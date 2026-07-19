"""Fail-closed physical-GPU scope for CUTLASS migration evidence tools."""

from __future__ import annotations

import argparse
import os

import torch


def add_target_gpu_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--expected-physical-gpu",
        type=int,
        choices=(4, 5),
        required=True,
        help=(
            "physical GPU exposed through CUDA_VISIBLE_DEVICES; CUTLASS migration "
            "evidence is restricted to GPU 4 or 5"
        ),
    )


def require_target_gpu(
    expected_physical_gpu: int | None = None,
) -> dict[str, object]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible not in {"4", "5"}:
        raise RuntimeError("set CUDA_VISIBLE_DEVICES to physical GPU 4 or 5")
    physical_gpu = int(visible)
    if expected_physical_gpu is not None:
        if expected_physical_gpu not in {4, 5}:
            raise RuntimeError("expected physical GPU must be 4 or 5")
        if physical_gpu != expected_physical_gpu:
            raise RuntimeError(
                "CUDA_VISIBLE_DEVICES does not match --expected-physical-gpu: "
                f"visible={physical_gpu} expected={expected_physical_gpu}"
            )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "benchmark requires exactly one CUDA-visible physical GPU; "
            f"found {torch.cuda.device_count()}"
        )
    logical_device = torch.cuda.current_device()
    if logical_device != 0:
        raise RuntimeError(
            "the single CUDA-visible physical GPU must map to logical device 0; "
            f"found {logical_device}"
        )
    capability = torch.cuda.get_device_capability(logical_device)
    if capability != (12, 0):
        raise RuntimeError(f"SM120 is required, found compute capability {capability}")
    properties = torch.cuda.get_device_properties(logical_device)
    return {
        "visible_devices": visible,
        "physical_index": physical_gpu,
        "logical_device": logical_device,
        "name": properties.name,
        "uuid": str(getattr(properties, "uuid", "")),
        "capability": list(capability),
    }
