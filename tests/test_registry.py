"""Registry contract: sparkinfer._OPS and the on-disk op directories stay in
lockstep, and every op honors the META/__all__ shape (invariant #3)."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
SPARKINFER_DIR = REPO / "sparkinfer"


def _on_disk_ops() -> list[str]:
    ops = []
    for group_dir in sorted(SPARKINFER_DIR.iterdir()):
        if not group_dir.is_dir() or group_dir.name.startswith("_"):
            continue
        for op_dir in sorted(group_dir.iterdir()):
            if not op_dir.is_dir() or op_dir.name.startswith("_"):
                continue
            ops.append(f"{group_dir.name}.{op_dir.name}")
    return ops


def _sparkinfer():
    return importlib.import_module("sparkinfer")


def test_registry_matches_disk():
    sparkinfer = _sparkinfer()
    overrides = set(sparkinfer._OP_MODULE_OVERRIDES)
    assert sorted(set(sparkinfer._OPS) - overrides) == _on_disk_ops(), (
        "sparkinfer._OPS and public op directories under sparkinfer/ must be "
        "in bijection apart from explicit private-module overrides"
    )
    for qualname, module_path in sparkinfer._OP_MODULE_OVERRIDES.items():
        assert qualname in sparkinfer._OPS
        module = importlib.import_module(f"sparkinfer.{module_path}")
        assert module.META.qualname == qualname


def test_list_ops_and_find_op():
    sparkinfer = _sparkinfer()
    metas = sparkinfer.list_ops()
    assert len(metas) == len(sparkinfer._OPS)
    for meta in metas:
        assert sparkinfer.find_op(meta.qualname) is meta
    with pytest.raises(KeyError):
        sparkinfer.find_op("no_such.op")


def test_every_op_meta_contract():
    sparkinfer = _sparkinfer()
    for meta in sparkinfer.list_ops():
        module = importlib.import_module(
            f"sparkinfer.{sparkinfer._op_module_path(meta.qualname)}"
        )
        assert isinstance(module.META, sparkinfer.OpMeta)
        assert set(module.__all__) == set(meta.entry_points) | {"META"}, meta.qualname
        assert any(
            name == "is_supported"
            or (name.startswith("is_") and name.endswith("_supported"))
            for name in meta.entry_points
        ), meta.qualname
        assert meta.archs and set(meta.archs) <= {"sm120a", "sm121a"}, meta.qualname
        assert meta.provenance.commit, f"{meta.qualname} missing provenance commit"
        assert meta.test_path and (REPO / meta.test_path).is_file(), (
            f"{meta.qualname} META.test_path {meta.test_path!r} does not exist"
        )


def test_clear_all_caches_never_forces_imports():
    sparkinfer = _sparkinfer()
    sparkinfer.clear_all_caches()  # must be a no-op / safe with nothing imported


def test_every_op_api_resolves():
    """Force-load every op's api and resolve every declared entry point.

    Catches facade/alias typos without a GPU; needs cutlass (kernel modules
    import it), so the CPU-only CI job skips this one.
    """
    pytest.importorskip("cutlass")
    sparkinfer = _sparkinfer()
    for meta in sparkinfer.list_ops():
        module = importlib.import_module(
            f"sparkinfer.{sparkinfer._op_module_path(meta.qualname)}"
        )
        for name in meta.entry_points:
            assert getattr(module, name) is not None, f"{meta.qualname}.{name}"
