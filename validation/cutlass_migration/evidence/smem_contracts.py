#!/usr/bin/env python3
"""Statically audit the CUTLASS DSL 4.6 shared-memory allocation contract.

The CUTLASS DSL 4.6 contract used by sparkinfer is intentionally narrow and covers
both production source and every retained benchmark/test kernel:

* every ``SmemAllocator`` constructor is called without arguments;
* the result is bound to one local name and receives exactly one reviewed
  public ``.allocate(...)`` or ``.allocate_tensor(...)`` call in the same
  lexical scope; and
* private ``_MemRange*`` implementation details are confined to
  ``sparkinfer/cute/smem.py``.

This tool parses source only.  It does not import sparkinfer, CUTLASS, Torch, or
CUDA, so it is safe to use as a pre-compilation migration gate.
"""

from __future__ import annotations

import argparse
import ast
import csv
import io
import json
import sys
import tempfile
import tokenize
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from validation.cutlass_migration.paths import REPO_ROOT


_SCHEMA = "sparkinfer.cute.smem_contracts.v1"
_CENTRAL_PRIVATE_MEMRANGE_PATH = Path("sparkinfer/cute/smem.py")
_AUDITED_SOURCE_ROOTS = {
    "production": (Path("sparkinfer"),),
    "infrastructure": (Path("benchmarks"), Path("tests"), Path("validation")),
}


@dataclass(frozen=True)
class _AllocationCall:
    method: str
    line: int
    column: int
    argument_count: int
    keyword_count: int
    argument_source: str
    typed: bool
    result_name: str
    result_bound_locally: bool


@dataclass
class _AllocatorRow:
    kind: str
    source_category: str
    path: str
    line: int
    column: int
    scope: str
    allocator_name: str
    constructor_argument_count: int
    constructor_keyword_count: int
    allocator_store_count: int = 0
    allocator_store_lines: list[int] = field(default_factory=list)
    allocation_count: int = 0
    allocation_methods: list[str] = field(default_factory=list)
    allocation_lines: list[int] = field(default_factory=list)
    allocation_argument_counts: list[int] = field(default_factory=list)
    allocation_keyword_counts: list[int] = field(default_factory=list)
    allocation_argument_sources: list[str] = field(default_factory=list)
    allocation_result_names: list[str] = field(default_factory=list)
    typed_allocation: bool = False
    allocation_after_constructor: bool = False
    allocation_result_bound_locally: bool = False
    violations: list[str] = field(default_factory=list)
    passed: bool = False


@dataclass
class _PrivateMemRangeRow:
    kind: str
    source_category: str
    path: str
    line: int
    column: int
    scope: str
    identifier: str
    centralized: bool
    violations: list[str]
    passed: bool


@dataclass
class _ParseErrorRow:
    kind: str
    source_category: str
    path: str
    line: int
    column: int
    scope: str
    message: str
    violations: list[str]
    passed: bool


def _is_smem_allocator_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    function = node.func
    return (
        isinstance(function, ast.Name) and function.id == "SmemAllocator"
    ) or (
        isinstance(function, ast.Attribute) and function.attr == "SmemAllocator"
    )


def _bound_name(node: ast.Call, parents: dict[ast.AST, ast.AST]) -> str | None:
    parent = parents.get(node)
    if isinstance(parent, ast.Assign) and parent.value is node:
        if len(parent.targets) == 1 and isinstance(parent.targets[0], ast.Name):
            return parent.targets[0].id
        return None
    if isinstance(parent, ast.AnnAssign) and parent.value is node:
        if isinstance(parent.target, ast.Name):
            return parent.target.id
        return None
    if isinstance(parent, ast.NamedExpr) and parent.value is node:
        if isinstance(parent.target, ast.Name):
            return parent.target.id
        return None
    return None


def _typed_allocation_expression(node: ast.AST | None) -> bool:
    """Recognize only the reviewed type-producing forms used by sparkinfer."""

    if isinstance(node, ast.Name):
        # CUTLASS struct types and local type aliases are class-style names:
        # SharedStorage, Storage, and the deliberately terse test alias S.
        return bool(node.id) and node.id[0].isupper()
    if isinstance(node, ast.Attribute):
        # DenseGemmKernel receives its reviewed cute.struct type via this
        # instance attribute.  Do not generalize arbitrary attributes into
        # type expressions: a scalar self.size must fail closed.
        return (
            node.attr == "shared_storage"
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
        )
    if isinstance(node, ast.Call):
        function = node.func
        terminal = (
            function.id
            if isinstance(function, ast.Name)
            else function.attr
            if isinstance(function, ast.Attribute)
            else ""
        )
        # These helpers return a cute.struct class; they do not allocate an
        # instance.  All current dynamic residual/indexer contracts use this
        # explicit suffix.
        return terminal.lower().endswith("storage_cls") and not any(
            isinstance(argument, ast.Starred) for argument in node.args
        )
    return False


def _allocate_tensor_contract(node: ast.Call) -> bool:
    """Recognize the reviewed public tensor-allocation spelling.

    CUTLASS 4.6's ``allocate_tensor`` is deliberately modeled separately from
    struct allocation: it takes no positional arguments, names all three
    contract fields, and uses a literal positive byte alignment.  Element type
    and layout may be local or attribute expressions because test/probe kernels
    legitimately specialize both through ``self``.
    """

    if node.args or any(keyword.arg is None for keyword in node.keywords):
        return False
    keywords = {keyword.arg: keyword.value for keyword in node.keywords}
    if len(keywords) != len(node.keywords) or set(keywords) != {
        "element_type",
        "layout",
        "byte_alignment",
    }:
        return False
    element_type = keywords["element_type"]
    layout = keywords["layout"]
    alignment = keywords["byte_alignment"]
    return (
        isinstance(element_type, (ast.Name, ast.Attribute))
        and isinstance(layout, (ast.Name, ast.Attribute, ast.Call))
        and isinstance(alignment, ast.Constant)
        and isinstance(alignment.value, int)
        and not isinstance(alignment.value, bool)
        and alignment.value > 0
    )


def _scope_nodes(scope: ast.AST) -> Iterable[ast.AST]:
    """Yield nodes in *scope* without descending into nested lexical scopes."""

    nested_scope_types = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
    stack = list(reversed(list(ast.iter_child_nodes(scope))))
    while stack:
        node = stack.pop()
        if isinstance(node, nested_scope_types):
            continue
        yield node
        stack.extend(reversed(list(ast.iter_child_nodes(node))))


def _all_scopes(tree: ast.Module) -> Iterable[tuple[ast.AST, str]]:
    yield tree, "<module>"

    def visit(node: ast.AST, prefix: str) -> Iterable[tuple[ast.AST, str]]:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                qualified = f"{prefix}.{child.name}" if prefix else child.name
                yield child, qualified
                yield from visit(child, qualified)
            elif isinstance(child, ast.Lambda):
                qualified = f"{prefix}.<lambda@{child.lineno}>" if prefix else f"<lambda@{child.lineno}>"
                yield child, qualified
                yield from visit(child, qualified)
            else:
                yield from visit(child, prefix)

    yield from visit(tree, "")


def _allocation_call(
    node: ast.Call,
    *,
    method: str,
    source: str,
    parents: dict[ast.AST, ast.AST],
) -> _AllocationCall:
    positional = list(node.args)
    expression = positional[0] if method == "allocate" and len(positional) == 1 else None
    typed = (
        expression is not None
        and not isinstance(expression, ast.Starred)
        and _typed_allocation_expression(expression)
        and not node.keywords
        if method == "allocate"
        else _allocate_tensor_contract(node)
    )
    argument_source = ast.get_source_segment(source, node)
    if argument_source is None:
        argument_source = ast.unparse(node)
    return _AllocationCall(
        method=method,
        line=node.lineno,
        column=node.col_offset,
        argument_count=len(positional),
        keyword_count=len(node.keywords),
        argument_source=argument_source or "",
        typed=typed,
        result_name=_bound_name(node, parents) or "",
        result_bound_locally=_bound_name(node, parents) is not None,
    )


def _audit_allocator_rows(
    path: str,
    source_category: str,
    source: str,
    tree: ast.Module,
) -> list[_AllocatorRow]:
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    rows: list[_AllocatorRow] = []
    for scope, scope_name in _all_scopes(tree):
        nodes = list(_scope_nodes(scope))
        constructors = [node for node in nodes if _is_smem_allocator_call(node)]
        bindings: dict[str, list[ast.Call]] = {}
        for constructor in constructors:
            assert isinstance(constructor, ast.Call)
            name = _bound_name(constructor, parents)
            if name is not None:
                bindings.setdefault(name, []).append(constructor)

        allocation_calls: dict[str, list[_AllocationCall]] = {}
        for node in nodes:
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr not in {"allocate", "allocate_tensor"} or not isinstance(
                node.func.value, ast.Name
            ):
                continue
            name = node.func.value.id
            if name in bindings:
                allocation_calls.setdefault(name, []).append(
                    _allocation_call(
                        node,
                        method=node.func.attr,
                        source=source,
                        parents=parents,
                    )
                )

        for constructor in constructors:
            assert isinstance(constructor, ast.Call)
            name = _bound_name(constructor, parents)
            calls = allocation_calls.get(name, []) if name is not None else []
            stores = [
                node
                for node in nodes
                if isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Store)
                and name is not None
                and node.id == name
            ]
            violations: list[str] = []
            if constructor.args or constructor.keywords:
                violations.append("constructor_not_zero_argument")
            if name is None:
                violations.append("allocator_not_bound_to_local_name")
            elif len(bindings[name]) != 1:
                violations.append("allocator_name_rebound_in_scope")
            if name is not None and len(stores) != 1:
                violations.append("allocator_store_count_not_one")
            if len(calls) != 1:
                violations.append("allocation_count_not_one")
            else:
                call = calls[0]
                if not call.typed:
                    violations.append(
                        "allocation_not_single_typed_argument"
                        if call.method == "allocate"
                        else "allocate_tensor_contract_invalid"
                    )
                if (call.line, call.column) <= (
                    constructor.lineno,
                    constructor.col_offset,
                ):
                    violations.append("allocation_not_after_constructor")
                if not call.result_bound_locally:
                    violations.append("allocation_result_not_bound_to_local_name")
            row = _AllocatorRow(
                kind="allocator",
                source_category=source_category,
                path=path,
                line=constructor.lineno,
                column=constructor.col_offset,
                scope=scope_name,
                allocator_name=name or "",
                constructor_argument_count=len(constructor.args),
                constructor_keyword_count=len(constructor.keywords),
                allocator_store_count=len(stores),
                allocator_store_lines=[node.lineno for node in stores],
                allocation_count=len(calls),
                allocation_methods=[call.method for call in calls],
                allocation_lines=[call.line for call in calls],
                allocation_argument_counts=[call.argument_count for call in calls],
                allocation_keyword_counts=[call.keyword_count for call in calls],
                allocation_argument_sources=[call.argument_source for call in calls],
                allocation_result_names=[call.result_name for call in calls],
                typed_allocation=len(calls) == 1 and calls[0].typed,
                allocation_after_constructor=(
                    len(calls) == 1
                    and (calls[0].line, calls[0].column)
                    > (constructor.lineno, constructor.col_offset)
                ),
                allocation_result_bound_locally=(
                    len(calls) == 1 and calls[0].result_bound_locally
                ),
                violations=violations,
                passed=not violations,
            )
            rows.append(row)
    return rows


def _private_memrange_rows(
    relative_path: Path,
    source_category: str,
    source: str,
) -> list[_PrivateMemRangeRow]:
    rows: list[_PrivateMemRangeRow] = []
    centralized = relative_path == _CENTRAL_PRIVATE_MEMRANGE_PATH
    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        for token in tokens:
            if token.type != tokenize.NAME or not token.string.startswith("_MemRange"):
                continue
            violations = [] if centralized else ["private_memrange_outside_central_module"]
            rows.append(
                _PrivateMemRangeRow(
                    kind="private_memrange",
                    source_category=source_category,
                    path=relative_path.as_posix(),
                    line=token.start[0],
                    column=token.start[1],
                    scope="",
                    identifier=token.string,
                    centralized=centralized,
                    violations=violations,
                    passed=not violations,
                )
            )
    except tokenize.TokenError:
        # ast.parse produces the canonical parse-error row below.
        pass
    return rows


def audit(root: Path) -> dict[str, object]:
    root = root.resolve()
    package = root / "sparkinfer"
    if not package.is_dir():
        raise ValueError(f"expected a sparkinfer package under audit root: {package}")

    allocator_rows: list[_AllocatorRow] = []
    private_rows: list[_PrivateMemRangeRow] = []
    parse_rows: list[_ParseErrorRow] = []
    categorized_python_files = sorted(
        (
            source_path,
            source_category,
        )
        for source_category, source_roots in _AUDITED_SOURCE_ROOTS.items()
        for source_root in source_roots
        for source_path in (root / source_root).rglob("*.py")
        if source_path.is_file()
    )
    for source_path, source_category in categorized_python_files:
        relative_path = source_path.relative_to(root)
        source = source_path.read_text(encoding="utf-8")
        private_rows.extend(
            _private_memrange_rows(relative_path, source_category, source)
        )
        try:
            tree = ast.parse(source, filename=relative_path.as_posix())
        except SyntaxError as exc:
            parse_rows.append(
                _ParseErrorRow(
                    kind="parse_error",
                    source_category=source_category,
                    path=relative_path.as_posix(),
                    line=exc.lineno or 0,
                    column=exc.offset or 0,
                    scope="",
                    message=exc.msg,
                    violations=["python_parse_error"],
                    passed=False,
                )
            )
            continue
        allocator_rows.extend(
            _audit_allocator_rows(
                relative_path.as_posix(), source_category, source, tree
            )
        )

    allocator_rows.sort(key=lambda row: (row.path, row.line, row.column))
    private_rows.sort(key=lambda row: (row.path, row.line, row.column))
    parse_rows.sort(key=lambda row: (row.path, row.line, row.column))
    rows = [
        *(asdict(row) for row in allocator_rows),
        *(asdict(row) for row in private_rows),
        *(asdict(row) for row in parse_rows),
    ]
    def counts_for(source_category: str | None) -> dict[str, int]:
        category_files = [
            path
            for path, category in categorized_python_files
            if source_category is None or category == source_category
        ]
        category_allocators = [
            row
            for row in allocator_rows
            if source_category is None or row.source_category == source_category
        ]
        category_private = [
            row
            for row in private_rows
            if source_category is None or row.source_category == source_category
        ]
        category_parse = [
            row
            for row in parse_rows
            if source_category is None or row.source_category == source_category
        ]
        category_rows = [
            row
            for row in rows
            if source_category is None or row["source_category"] == source_category
        ]
        return {
            "python_file_count": len(category_files),
            "allocator_count": len(category_allocators),
            "allocator_pass_count": sum(row.passed for row in category_allocators),
            "allocator_fail_count": sum(not row.passed for row in category_allocators),
            "allocation_call_count": sum(
                row.allocation_count for row in category_allocators
            ),
            "allocate_call_count": sum(
                row.allocation_methods.count("allocate") for row in category_allocators
            ),
            "allocate_tensor_call_count": sum(
                row.allocation_methods.count("allocate_tensor")
                for row in category_allocators
            ),
            "private_memrange_identifier_count": len(category_private),
            "private_memrange_centralized_count": sum(
                row.centralized for row in category_private
            ),
            "private_memrange_outside_count": sum(
                not row.centralized for row in category_private
            ),
            "parse_error_count": len(category_parse),
            "violation_count": sum(
                len(row["violations"]) for row in category_rows
            ),
            "row_count": len(category_rows),
        }

    counts = counts_for(None)
    source_counts = {
        category: counts_for(category) for category in _AUDITED_SOURCE_ROOTS
    }
    return {
        "schema": _SCHEMA,
        "root": str(root),
        "audited_source_roots": {
            category: [path.as_posix() for path in roots]
            for category, roots in _AUDITED_SOURCE_ROOTS.items()
        },
        "central_private_memrange_path": _CENTRAL_PRIVATE_MEMRANGE_PATH.as_posix(),
        "rows": rows,
        "counts": counts,
        "source_counts": source_counts,
        "passed": counts["violation_count"] == 0,
    }


def _csv_text(report: dict[str, object]) -> str:
    rows = report["rows"]
    assert isinstance(rows, list)
    counts = report["counts"]
    assert isinstance(counts, dict)
    normalized: list[dict[str, object]] = []
    for row in rows:
        assert isinstance(row, dict)
        normalized.append(
            {
                **row,
                "violations": json.dumps(row.get("violations", []), separators=(",", ":")),
                "allocation_lines": json.dumps(row.get("allocation_lines", []), separators=(",", ":")),
                "allocation_argument_sources": json.dumps(
                    row.get("allocation_argument_sources", []), separators=(",", ":")
                ),
                "allocator_store_lines": json.dumps(
                    row.get("allocator_store_lines", []), separators=(",", ":")
                ),
                "allocation_result_names": json.dumps(
                    row.get("allocation_result_names", []), separators=(",", ":")
                ),
                "allocation_methods": json.dumps(
                    row.get("allocation_methods", []), separators=(",", ":")
                ),
                "allocation_argument_counts": json.dumps(
                    row.get("allocation_argument_counts", []), separators=(",", ":")
                ),
                "allocation_keyword_counts": json.dumps(
                    row.get("allocation_keyword_counts", []), separators=(",", ":")
                ),
            }
        )
    normalized.append(
        {
            "kind": "summary",
            "path": "",
            "line": 0,
            "column": 0,
            "scope": "",
            "passed": report["passed"],
            "counts_json": json.dumps(counts, sort_keys=True, separators=(",", ":")),
        }
    )
    fields = sorted({key for row in normalized for key in row})
    priority = ["kind", "path", "line", "column", "scope", "passed", "violations"]
    fields = [*priority, *(field for field in fields if field not in priority)]
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(normalized)
    return stream.getvalue()


def _write_fixture(root: Path, relative: str, source: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def _violation_codes(report: dict[str, object]) -> set[str]:
    rows = report["rows"]
    assert isinstance(rows, list)
    return {
        code
        for row in rows
        if isinstance(row, dict)
        for code in row.get("violations", [])
    }


def self_test() -> dict[str, object]:
    cases: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="sparkinfer-smem-contract-selftest-") as raw:
        base = Path(raw)

        positive = base / "positive"
        _write_fixture(
            positive,
            "sparkinfer/good.py",
            "def kernel():\n"
            "    smem = cutlass.utils.SmemAllocator()\n"
            "    storage = smem.allocate(SharedStorage)\n",
        )
        _write_fixture(
            positive,
            "sparkinfer/cute/smem.py",
            "def bridge():\n    return cute.struct._MemRangeData\n",
        )
        _write_fixture(
            positive,
            "benchmarks/probe.py",
            "def kernel():\n"
            "    smem = cutlass.utils.SmemAllocator()\n"
            "    storage = smem.allocate(Storage)\n",
        )
        _write_fixture(
            positive,
            "tests/test_tensor_probe.py",
            "def kernel(self):\n"
            "    s_layout = cute.make_layout((16, 16))\n"
            "    smem = cutlass.utils.SmemAllocator()\n"
            "    tensor = smem.allocate_tensor(\n"
            "        element_type=self.dtype, layout=s_layout, byte_alignment=128\n"
            "    )\n",
        )
        report = audit(positive)
        ok = bool(report["passed"]) and report["counts"] == {
            "python_file_count": 4,
            "allocator_count": 3,
            "allocator_pass_count": 3,
            "allocator_fail_count": 0,
            "allocation_call_count": 3,
            "allocate_call_count": 2,
            "allocate_tensor_call_count": 1,
            "private_memrange_identifier_count": 1,
            "private_memrange_centralized_count": 1,
            "private_memrange_outside_count": 0,
            "parse_error_count": 0,
            "violation_count": 0,
            "row_count": 4,
        } and report["source_counts"] == {
            "production": {
                "python_file_count": 2,
                "allocator_count": 1,
                "allocator_pass_count": 1,
                "allocator_fail_count": 0,
                "allocation_call_count": 1,
                "allocate_call_count": 1,
                "allocate_tensor_call_count": 0,
                "private_memrange_identifier_count": 1,
                "private_memrange_centralized_count": 1,
                "private_memrange_outside_count": 0,
                "parse_error_count": 0,
                "violation_count": 0,
                "row_count": 2,
            },
            "infrastructure": {
                "python_file_count": 2,
                "allocator_count": 2,
                "allocator_pass_count": 2,
                "allocator_fail_count": 0,
                "allocation_call_count": 2,
                "allocate_call_count": 1,
                "allocate_tensor_call_count": 1,
                "private_memrange_identifier_count": 0,
                "private_memrange_centralized_count": 0,
                "private_memrange_outside_count": 0,
                "parse_error_count": 0,
                "violation_count": 0,
                "row_count": 2,
            },
        }
        cases.append({"name": "positive", "passed": ok})

        negative_sources = {
            "constructor_argument": (
                "def kernel():\n"
                "    smem = cutlass.utils.SmemAllocator(64)\n"
                "    storage = smem.allocate(Storage)\n",
                "constructor_not_zero_argument",
            ),
            "missing_allocate": (
                "def kernel():\n    smem = cutlass.utils.SmemAllocator()\n",
                "allocation_count_not_one",
            ),
            "double_allocate": (
                "def kernel():\n"
                "    smem = cutlass.utils.SmemAllocator()\n"
                "    smem.allocate(A)\n"
                "    smem.allocate(B)\n",
                "allocation_count_not_one",
            ),
            "untyped_allocate": (
                "def kernel():\n"
                "    smem = cutlass.utils.SmemAllocator()\n"
                "    storage = smem.allocate(4096)\n",
                "allocation_not_single_typed_argument",
            ),
            "scalar_name_allocate": (
                "def kernel():\n"
                "    size = 4096\n"
                "    smem = cutlass.utils.SmemAllocator()\n"
                "    storage = smem.allocate(size)\n",
                "allocation_not_single_typed_argument",
            ),
            "allocator_rebinding": (
                "def kernel():\n"
                "    smem = cutlass.utils.SmemAllocator()\n"
                "    smem = replacement\n"
                "    storage = smem.allocate(Storage)\n",
                "allocator_store_count_not_one",
            ),
            "pre_constructor_allocate": (
                "def kernel():\n"
                "    storage = smem.allocate(Storage)\n"
                "    smem = cutlass.utils.SmemAllocator()\n",
                "allocation_not_after_constructor",
            ),
            "discarded_allocate_result": (
                "def kernel():\n"
                "    smem = cutlass.utils.SmemAllocator()\n"
                "    smem.allocate(Storage)\n",
                "allocation_result_not_bound_to_local_name",
            ),
            "unbound_allocator": (
                "def kernel():\n    cutlass.utils.SmemAllocator()\n",
                "allocator_not_bound_to_local_name",
            ),
            "private_outside": (
                "def bridge():\n    return cute.struct._MemRangeData\n",
                "private_memrange_outside_central_module",
            ),
            "allocate_tensor_missing_alignment": (
                "def kernel(self):\n"
                "    smem = cutlass.utils.SmemAllocator()\n"
                "    tensor = smem.allocate_tensor(\n"
                "        element_type=self.dtype, layout=s_layout\n"
                "    )\n",
                "allocate_tensor_contract_invalid",
            ),
            "allocate_tensor_positional": (
                "def kernel(self):\n"
                "    smem = cutlass.utils.SmemAllocator()\n"
                "    tensor = smem.allocate_tensor(self.dtype, s_layout, 128)\n",
                "allocate_tensor_contract_invalid",
            ),
            "allocate_tensor_invalid_alignment": (
                "def kernel(self):\n"
                "    smem = cutlass.utils.SmemAllocator()\n"
                "    tensor = smem.allocate_tensor(\n"
                "        element_type=self.dtype, layout=s_layout, byte_alignment=0\n"
                "    )\n",
                "allocate_tensor_contract_invalid",
            ),
        }
        for name, (source, expected) in negative_sources.items():
            case_root = base / name
            _write_fixture(case_root, "sparkinfer/cute/smem.py", "# centralized bridge\n")
            _write_fixture(case_root, "sparkinfer/case.py", source)
            case_report = audit(case_root)
            codes = _violation_codes(case_report)
            cases.append(
                {
                    "name": name,
                    "expected_violation": expected,
                    "observed_violations": sorted(codes),
                    "passed": not case_report["passed"] and expected in codes,
                }
            )

    passed = all(bool(case["passed"]) for case in cases)
    return {
        "schema": f"{_SCHEMA}.self_test",
        "cases": cases,
        "case_count": len(cases),
        "passed": passed,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT,
        help="repository root containing sparkinfer (default: script repository)",
    )
    parser.add_argument("--format", choices=("json", "csv"), default="json")
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run built-in positive and negative static fixtures",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        report = self_test() if arguments.self_test else audit(arguments.root)
    except (OSError, ValueError) as exc:
        print(f"audit_cute_smem_contracts.py: error: {exc}", file=sys.stderr)
        return 2
    if arguments.self_test and arguments.format == "csv":
        print("audit_cute_smem_contracts.py: --self-test requires JSON output", file=sys.stderr)
        return 2
    text = (
        json.dumps(report, indent=2, sort_keys=True) + "\n"
        if arguments.format == "json"
        else _csv_text(report)
    )
    if arguments.output is None:
        sys.stdout.write(text)
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(text, encoding="utf-8")
    return 0 if bool(report["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
