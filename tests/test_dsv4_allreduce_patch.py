from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def _load_ar_module():
    module_name = "_test_dsv4_allreduce_patch"
    if module_name in sys.modules:
        return sys.modules[module_name]

    pcie_stub = type(sys)("b12x.distributed.pcie_oneshot")
    pcie_stub.PCIeOneshotAllReduce = object
    pcie_stub.SUPPORTED_WORLD_SIZES = (2, 4, 6, 8)
    sys.modules.setdefault("b12x.distributed.pcie_oneshot", pcie_stub)

    spec = importlib.util.spec_from_file_location(
        module_name,
        Path(__file__).resolve().parents[1] / "b12x" / "integration" / "dsv4_allreduce_patch.py",
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


ar = _load_ar_module()


class _FakeTensor:
    def __init__(
        self,
        shape: tuple[int, ...],
        *,
        dtype: torch.dtype = torch.bfloat16,
        device_type: str = "cuda",
        contiguous: bool = True,
        element_size: int = 2,
    ):
        self.shape = shape
        self.dtype = dtype
        self.device = SimpleNamespace(type=device_type)
        self._contiguous = contiguous
        self._element_size = element_size

    def is_contiguous(self) -> bool:
        return self._contiguous

    def element_size(self) -> int:
        return self._element_size


@pytest.fixture(autouse=True)
def _clear_ar_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("B12X_PCIE_AR_MAX_TOKENS", raising=False)
    monkeypatch.delenv("B12X_PCIE_AR_PREWARM_SHAPES", raising=False)


@pytest.mark.parametrize("tokens", [1, 4, 5, 8, 16, 32])
def test_fast_path_keeps_verified_small_t_buckets(tokens: int) -> None:
    assert ar._is_fast_path(_FakeTensor((tokens, 4096)))


def test_fast_path_rejects_t64_regression_bucket_by_default() -> None:
    assert not ar._is_fast_path(_FakeTensor((64, 4096)))


def test_max_tokens_env_can_narrow_but_not_broaden_to_t64(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("B12X_PCIE_AR_MAX_TOKENS", "8")
    assert ar._is_fast_path(_FakeTensor((8, 4096)))
    assert not ar._is_fast_path(_FakeTensor((16, 4096)))

    monkeypatch.setenv("B12X_PCIE_AR_MAX_TOKENS", "64")
    assert ar._is_fast_path(_FakeTensor((32, 4096)))
    assert not ar._is_fast_path(_FakeTensor((64, 4096)))


def test_fast_path_rejects_non_dsv4_shapes_and_dtypes() -> None:
    assert not ar._is_fast_path(_FakeTensor((32, 2048)))
    assert not ar._is_fast_path(_FakeTensor((32, 4096), dtype=torch.float16))
    assert not ar._is_fast_path(_FakeTensor((32, 4096), device_type="cpu"))
    assert not ar._is_fast_path(_FakeTensor((32, 4096), contiguous=False))


def test_capture_prewarm_shapes_filter_to_fast_path() -> None:
    shapes = ar._capture_prewarm_shapes("4,4096;32,4096;64,4096;1,2048;bad")
    assert shapes == [(1, 4096), (4, 4096), (32, 4096)]


def test_default_capture_prewarm_includes_t32_and_excludes_t64() -> None:
    shapes = ar._capture_prewarm_shapes(None)
    assert (32, 4096) in shapes
    assert (64, 4096) not in shapes
