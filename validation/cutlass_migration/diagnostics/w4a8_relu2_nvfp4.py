"""GPU-only isolation for the W4A8 NVFP4/ReLU2 small-tile cliff."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
from pathlib import Path
import sys
from typing import Any

import torch

from validation.cutlass_migration.core.gpu_scope import (
    add_target_gpu_argument,
    require_target_gpu,
)


def _prepare_cutlass_runtime() -> str:
    version = importlib.metadata.version("nvidia-cutlass-dsl")
    if version.startswith("4.5."):
        from cutlass.base_dsl import _mlir_helpers

        sys.modules["cutlass._mlir_helpers"] = _mlir_helpers

    from sparkinfer.cute.runtime_patches import apply_cutlass_runtime_patches

    apply_cutlass_runtime_patches()
    return version


def _metrics(output: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    got = output.float()
    ref = reference.float()
    error = got - ref
    row_cos = torch.nn.functional.cosine_similarity(got, ref, dim=1)
    row_rmse = error.square().mean(dim=1).sqrt()
    row_max = error.abs().amax(dim=1)
    worst_cos = int(row_cos.argmin().item())
    worst_rmse = int(row_rmse.argmax().item())
    worst_max = int(row_max.argmax().item())
    column_tiles = []
    for begin in range(0, got.shape[1], 128):
        end = min(begin + 128, got.shape[1])
        got_tile = got[:, begin:end]
        ref_tile = ref[:, begin:end]
        tile_error = got_tile - ref_tile
        column_tiles.append(
            {
                "begin": begin,
                "end": end,
                "cos": float(
                    torch.nn.functional.cosine_similarity(
                        got_tile.flatten(), ref_tile.flatten(), dim=0
                    ).item()
                ),
                "rmse": float(tile_error.square().mean().sqrt().item()),
                "max_abs": float(tile_error.abs().max().item()),
                "output_norm": float(got_tile.norm().item()),
                "reference_norm": float(ref_tile.norm().item()),
            }
        )
    return {
        "shape": list(got.shape),
        "finite": bool(torch.isfinite(got).all().item()),
        "nonzero_count": int(torch.count_nonzero(got).item()),
        "output_norm": float(got.norm().item()),
        "reference_norm": float(ref.norm().item()),
        "cos": float(row_cos.mean().item()),
        "global_cos": float(
            torch.nn.functional.cosine_similarity(
                got.flatten(), ref.flatten(), dim=0
            ).item()
        ),
        "rmse": float(error.square().mean().sqrt().item()),
        "mean_abs": float(error.abs().mean().item()),
        "max_abs": float(error.abs().max().item()),
        "row_cos_below_0_999": int((row_cos < 0.999).sum().item()),
        "row_cos_below_0_99": int((row_cos < 0.99).sum().item()),
        "row_cos_min": float(row_cos[worst_cos].item()),
        "row_cos_min_index": worst_cos,
        "row_rmse_max": float(row_rmse[worst_rmse].item()),
        "row_rmse_max_index": worst_rmse,
        "row_max_abs": float(row_max[worst_max].item()),
        "row_max_abs_index": worst_max,
        "column_tiles": column_tiles,
        "per_row": [
            {
                "row": row,
                "cos": float(row_cos[row].item()),
                "rmse": float(row_rmse[row].item()),
                "max_abs": float(row_max[row].item()),
            }
            for row in range(got.shape[0])
        ],
    }


def _region_metrics(
    output: torch.Tensor,
    reference: torch.Tensor,
    begin: int,
    end: int,
) -> dict[str, float] | None:
    if begin >= end:
        return None
    got = output[begin:end].float()
    ref = reference[begin:end].float()
    error = got - ref
    return {
        "begin": begin,
        "end": end,
        "cos": float(
            torch.nn.functional.cosine_similarity(
                got.flatten(), ref.flatten(), dim=0
            ).item()
        ),
        "rmse": float(error.square().mean().sqrt().item()),
        "max_abs": float(error.abs().max().item()),
    }


def _run_route_isolation(
    run_w4a8_dynamic: Any,
    *,
    tile_m: int,
    m: int,
    slot: int,
) -> dict[str, Any]:
    output, reference, debug = run_w4a8_dynamic(
        recipe="w4a8_nvfp4",
        activation="relu2",
        E=4,
        m=m,
        K=256,
        n=128,
        top_k=2,
        seed=1_000 + tile_m,
        tile_m=tile_m,
        route_weight_slot=slot,
        return_debug=True,
    )
    return {
        "slot": slot,
        "metrics": _metrics(output, reference),
        "reference_variant_metrics": {
            key.removeprefix("reference_"): _metrics(output, value)
            for key, value in debug.items()
            if key.startswith("reference_")
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_target_gpu_argument(parser)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tiles", type=int, nargs="+", default=[16, 32, 64])
    args = parser.parse_args()

    gpu_scope = require_target_gpu(args.expected_physical_gpu)
    cutlass_version = _prepare_cutlass_runtime()

    from tests.test_w4a8_dynamic_kernel import _run_w4a8_dynamic

    cases = []
    for tile_m in args.tiles:
        m = {16: 1, 32: 33, 64: 65, 128: 129}[tile_m]
        output, reference, debug = _run_w4a8_dynamic(
            recipe="w4a8_nvfp4",
            activation="relu2",
            E=4,
            m=m,
            K=256,
            n=128,
            top_k=2,
            seed=1_000 + tile_m,
            tile_m=tile_m,
            return_debug=True,
        )
        full_end = math.floor(m / tile_m) * tile_m
        cases.append(
            {
                "tile_m": tile_m,
                "m": m,
                "metrics": _metrics(output, reference),
                "complete_tile_rows": _region_metrics(
                    output, reference, 0, full_end
                ),
                "tail_rows": _region_metrics(output, reference, full_end, m),
                "routing": {
                    key: value.tolist()
                    for key, value in debug.items()
                    if not key.startswith("reference_")
                },
                "reference_variant_metrics": {
                    key.removeprefix("reference_"): _metrics(output, value)
                    for key, value in debug.items()
                    if key.startswith("reference_")
                },
                "route_isolation": [
                    _run_route_isolation(
                        _run_w4a8_dynamic,
                        tile_m=tile_m,
                        m=m,
                        slot=slot,
                    )
                    for slot in range(2)
                ],
            }
        )

    payload = {
        "schema": "sparkinfer.w4a8.relu2_nvfp4_diagnostic.v1",
        "toolchain": {
            "cutlass_dsl": cutlass_version,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
        "gpu": {
            "cuda_visible_devices": gpu_scope["visible_devices"],
            "physical_index": gpu_scope["physical_index"],
            "name": gpu_scope["name"],
            "uuid": gpu_scope["uuid"],
            "capability": gpu_scope["capability"],
            "visible_ordinal": gpu_scope["logical_device"],
        },
        "cases": cases,
    }
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
