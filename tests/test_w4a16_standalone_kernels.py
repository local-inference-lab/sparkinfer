from __future__ import annotations

import os
import sys

import pytest
import torch

from validation.cutlass_migration.diagnostics import w4a16_standalone


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
def test_w4a16_standalone_gemm_and_activation_match_gpu_oracles_under_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_physical_gpu = os.environ.get("CUDA_VISIBLE_DEVICES")
    if expected_physical_gpu not in {"4", "5"}:
        pytest.fail("set CUDA_VISIBLE_DEVICES to physical GPU 4 or 5")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validation.cutlass_migration.diagnostics.w4a16_standalone",
            "--expected-physical-gpu",
            expected_physical_gpu,
            "--device",
            str(torch.cuda.current_device()),
            "--warmup",
            "1",
            "--iterations",
            "1",
            "--repeats",
            "1",
            "--label",
            "pytest-graph-oracle",
        ],
    )

    w4a16_standalone.main()
