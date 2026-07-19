#!/usr/bin/env python3
"""Verify source-level CuTe kernel coverage against GPU-produced audit CSVs."""

from __future__ import annotations

import argparse
import ast
import csv
import fnmatch
import re
from dataclasses import dataclass
from pathlib import Path

from validation.cutlass_migration.paths import DATA_ROOT, REPO_ROOT


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class SourceKernel:
    path: str
    qualified_name: str


@dataclass(frozen=True)
class InventoryEntry:
    status: str
    path: str
    qualified_name: str
    kernel_symbol_glob: str
    reason: str

    @property
    def source_kernel(self) -> SourceKernel:
        return SourceKernel(self.path, self.qualified_name)


def _is_cute_kernel(decorator: ast.expr) -> bool:
    return bool(
        isinstance(decorator, ast.Attribute)
        and decorator.attr == "kernel"
        and isinstance(decorator.value, ast.Name)
        and decorator.value.id == "cute"
    )


def _source_kernels(root: Path) -> set[SourceKernel]:
    kernels: set[SourceKernel] = set()
    for path in sorted((root / "b12x").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        names: list[str] = []

        class Visitor(ast.NodeVisitor):
            def visit_ClassDef(self, node: ast.ClassDef) -> None:
                names.append(node.name)
                self.generic_visit(node)
                names.pop()

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                if any(_is_cute_kernel(item) for item in node.decorator_list):
                    kernels.add(
                        SourceKernel(
                            str(path.relative_to(root)),
                            ".".join((*names, node.name)),
                        )
                    )
                names.append(node.name)
                self.generic_visit(node)
                names.pop()

            visit_AsyncFunctionDef = visit_FunctionDef

        Visitor().visit(tree)
    return kernels


def _read_inventory(path: Path) -> list[InventoryEntry]:
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source, delimiter="\t")
        expected = {
            "status",
            "path",
            "qualified_name",
            "kernel_symbol_glob",
            "reason",
        }
        if set(reader.fieldnames or ()) != expected:
            raise ValueError(f"unexpected inventory columns: {reader.fieldnames}")
        return [InventoryEntry(**row) for row in reader]


def _read_audited_symbols(paths: list[Path]) -> set[str]:
    symbols: set[str] = set()
    for path in paths:
        identities: set[tuple[str, str]] = set()
        with path.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            required = {
                "manifest_status",
                "semantic_key",
                "comparison_semantic_key",
                "kernel",
            }
            missing = sorted(required - set(reader.fieldnames or ()))
            if missing:
                raise ValueError(f"{path}: missing resource columns {missing}")
            for row_number, row in enumerate(reader, start=2):
                semantic_key = str(row.get("semantic_key", ""))
                comparison_key = str(row.get("comparison_semantic_key", ""))
                kernel = str(row.get("kernel", ""))
                if row.get("manifest_status") != "ok":
                    raise ValueError(f"{path}:{row_number}: invalid manifest status")
                if not _SHA256_RE.fullmatch(semantic_key):
                    raise ValueError(f"{path}:{row_number}: invalid semantic_key")
                if not _SHA256_RE.fullmatch(comparison_key):
                    raise ValueError(
                        f"{path}:{row_number}: invalid comparison_semantic_key"
                    )
                if not kernel:
                    raise ValueError(f"{path}:{row_number}: empty CUDA symbol")
                identity = (comparison_key, kernel)
                if identity in identities:
                    raise ValueError(
                        f"{path}:{row_number}: duplicate comparison/symbol row"
                    )
                identities.add(identity)
                symbols.add(kernel)
    return symbols


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("resource_reports", nargs="+", type=Path)
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT,
        help="repository root containing b12x/",
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        default=DATA_ROOT / "cute_kernel_source_inventory.tsv",
    )
    args = parser.parse_args()

    try:
        inventory = _read_inventory(args.inventory)
        source_kernels = _source_kernels(args.root)
        audited_symbols = _read_audited_symbols(args.resource_reports)
    except (OSError, SyntaxError, ValueError) as exc:
        parser.error(str(exc))

    inventory_keys = [entry.source_kernel for entry in inventory]
    duplicate_keys = sorted(
        {key for key in inventory_keys if inventory_keys.count(key) > 1},
        key=lambda item: (item.path, item.qualified_name),
    )
    missing_inventory = sorted(
        source_kernels - set(inventory_keys),
        key=lambda item: (item.path, item.qualified_name),
    )
    stale_inventory = sorted(
        set(inventory_keys) - source_kernels,
        key=lambda item: (item.path, item.qualified_name),
    )
    invalid_entries: list[tuple[InventoryEntry, str]] = []
    missing_audited: list[InventoryEntry] = []
    for entry in inventory:
        if entry.status == "production":
            if not entry.kernel_symbol_glob or entry.reason:
                invalid_entries.append(
                    (
                        entry,
                        "production entries require a symbol glob and no exclusion reason",
                    )
                )
                continue
        elif entry.status == "diagnostic":
            if not entry.kernel_symbol_glob or not entry.reason:
                invalid_entries.append(
                    (entry, "diagnostic entries require a symbol glob and reason")
                )
        elif entry.status == "archaeology":
            if (
                entry.kernel_symbol_glob
                or not entry.reason
                or "/legacy/" not in entry.path
            ):
                invalid_entries.append(
                    (
                        entry,
                        "archaeology entries require a legacy path, no symbol glob, "
                        "and an exclusion reason",
                    )
                )
        else:
            invalid_entries.append((entry, f"unknown status {entry.status!r}"))
        if entry.kernel_symbol_glob and not any(
            fnmatch.fnmatchcase(symbol, entry.kernel_symbol_glob)
            for symbol in audited_symbols
        ):
            missing_audited.append(entry)

    symbol_inventory_matches = {
        symbol: [
            entry
            for entry in inventory
            if entry.kernel_symbol_glob
            and fnmatch.fnmatchcase(symbol, entry.kernel_symbol_glob)
        ]
        for symbol in audited_symbols
    }
    unowned_audited_symbols = sorted(
        symbol for symbol, matches in symbol_inventory_matches.items() if not matches
    )
    multiply_owned_audited_symbols = {
        symbol: matches
        for symbol, matches in symbol_inventory_matches.items()
        if len(matches) > 1
    }

    for key in duplicate_keys:
        print(f"error: duplicate inventory entry: {key.path}:{key.qualified_name}")
    for key in missing_inventory:
        print(
            f"error: source kernel missing from inventory: {key.path}:{key.qualified_name}"
        )
    for key in stale_inventory:
        print(f"error: stale inventory entry: {key.path}:{key.qualified_name}")
    for entry, reason in invalid_entries:
        print(
            f"error: invalid inventory entry {entry.path}:{entry.qualified_name}: {reason}"
        )
    for entry in missing_audited:
        print(
            "error: source kernel has no audited cubin entry: "
            f"{entry.path}:{entry.qualified_name} ({entry.kernel_symbol_glob})"
        )
    for symbol in unowned_audited_symbols:
        print(f"error: audited cubin entry has no source inventory owner: {symbol}")
    for symbol, matches in sorted(multiply_owned_audited_symbols.items()):
        owners = [f"{entry.path}:{entry.qualified_name}" for entry in matches]
        print(
            "error: audited cubin entry has multiple source inventory owners: "
            f"{symbol} ({owners!r})"
        )
    production_count = sum(entry.status == "production" for entry in inventory)
    diagnostic_count = sum(entry.status == "diagnostic" for entry in inventory)
    archaeology_count = sum(entry.status == "archaeology" for entry in inventory)
    print(
        f"source_kernels={len(source_kernels)} inventory={len(inventory)} "
        f"production={production_count} diagnostic={diagnostic_count} "
        f"archaeology={archaeology_count} "
        f"audited_symbols={len(audited_symbols)} "
        f"missing_audited={len(missing_audited)} "
        f"unowned_audited={len(unowned_audited_symbols)} "
        f"multiply_owned_audited={len(multiply_owned_audited_symbols)}"
    )
    return int(
        bool(
            duplicate_keys
            or missing_inventory
            or stale_inventory
            or invalid_entries
            or missing_audited
            or unowned_audited_symbols
            or multiply_owned_audited_symbols
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
