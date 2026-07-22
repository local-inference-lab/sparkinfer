from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


ROOT = Path(__file__).parents[1]
PCIE_PACKAGE = "sparkinfer.comm.pcie"
RUNTIME_CUDA_SOURCES = {
    "pcie_dcp_a2a.cu",
    "pcie_dma.cu",
    "pcie_oneshot.cu",
    "pcie_twoshot.cu",
}


def test_runtime_cuda_sources_are_in_package_data() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    package_data = config["tool"]["setuptools"]["package-data"]

    assert package_data[PCIE_PACKAGE] == ["*.cu"]
    assert {
        path.name for path in (ROOT / "sparkinfer" / "comm" / "pcie").glob("*.cu")
    } == RUNTIME_CUDA_SOURCES
