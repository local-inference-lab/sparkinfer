"""Capture pre-ReLU W4A8 FC1 accumulators for the small-tile NVFP4 path."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
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


def _row_metrics(got: torch.Tensor, ref: torch.Tensor) -> tuple[torch.Tensor, ...]:
    got = got.float()
    ref = ref.float()
    error = got - ref
    dot = (got * ref).sum(dim=1)
    denom = got.norm(dim=1) * ref.norm(dim=1)
    cos = torch.where(denom > 0, dot / denom, torch.zeros_like(dot))
    rmse = error.square().mean(dim=1).sqrt()
    max_abs = error.abs().amax(dim=1)
    return cos, rmse, max_abs


def _route_slot(topk_ids: torch.Tensor, token: int, expert: int) -> int:
    match = torch.nonzero(topk_ids[token] == expert, as_tuple=False).flatten()
    if match.numel() != 1:
        raise RuntimeError(
            f"expected one route for token={token}, expert={expert}; got {match.tolist()}"
        )
    return int(match[0].item())


def _case(run: Any, tile_m: int) -> dict[str, Any]:
    m = {16: 16, 32: 33}[tile_m]
    output, reference, debug = run(
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
        capture_fc1_raw=True,
    )
    del output, reference
    expert = debug["fc1_raw_physical_expert"]
    expert_local = debug["fc1_raw_physical_expert_local"]
    valid = expert >= 0
    physical_rows = torch.nonzero(valid, as_tuple=False).flatten()
    got = debug["fc1_raw_device"][valid]
    ref = debug["fc1_raw_reference"][valid]
    cos, rmse, max_abs = _row_metrics(got, ref)

    # Cross-row cosine checks whether a bad packed row is merely permuted.
    normed_got = torch.nn.functional.normalize(got.float(), dim=1)
    normed_ref = torch.nn.functional.normalize(ref.float(), dim=1)
    cross_cos = normed_got @ normed_ref.T
    best_cos, best_row = cross_cos.max(dim=1)

    per_row = []
    for idx, physical_row_tensor in enumerate(physical_rows):
        physical_row = int(physical_row_tensor.item())
        token = int(debug["token_map"][physical_row].item())
        eid = int(expert[physical_row].item())
        local = int(expert_local[physical_row].item())
        per_row.append(
            {
                "physical_row": physical_row,
                "token": token,
                "expert": eid,
                "expert_local": local,
                "expert_local_mod_tile": local % tile_m,
                "route_slot": _route_slot(debug["topk_ids"], token, eid),
                "cos": float(cos[idx].item()),
                "rmse": float(rmse[idx].item()),
                "max_abs": float(max_abs[idx].item()),
                "best_cross_cos": float(best_cos[idx].item()),
                "best_cross_physical_row": int(physical_rows[best_row[idx]].item()),
                "device": got[idx].tolist(),
                "reference": ref[idx].tolist(),
            }
        )

    bands = []
    for name, predicate in (
        ("local_lt_16", lambda local: local % tile_m < 16),
        ("local_ge_16", lambda local: local % tile_m >= 16),
    ):
        indices = [
            idx
            for idx, row in enumerate(per_row)
            if predicate(int(row["expert_local"]))
        ]
        if not indices:
            continue
        ii = torch.tensor(indices, dtype=torch.long)
        bands.append(
            {
                "name": name,
                "rows": len(indices),
                "cos_mean": float(cos[ii].mean().item()),
                "cos_min": float(cos[ii].min().item()),
                "rmse_mean": float(rmse[ii].mean().item()),
                "rmse_max": float(rmse[ii].max().item()),
                "max_abs": float(max_abs[ii].max().item()),
            }
        )

    return {
        "tile_m": tile_m,
        "m": m,
        "rows": len(per_row),
        "cos_mean": float(cos.mean().item()),
        "cos_min": float(cos.min().item()),
        "rmse_mean": float(rmse.mean().item()),
        "rmse_max": float(rmse.max().item()),
        "max_abs": float(max_abs.max().item()),
        "bad_row_count": int((cos < 0.999).sum().item()),
        "bands": bands,
        "per_row": per_row,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_target_gpu_argument(parser)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tiles", type=int, nargs="+", default=[16, 32])
    args = parser.parse_args()

    gpu_scope = require_target_gpu(args.expected_physical_gpu)
    version = _prepare_cutlass_runtime()
    from tests.test_w4a8_dynamic_kernel import _run_w4a8_dynamic

    cases = [_case(_run_w4a8_dynamic, tile_m) for tile_m in args.tiles]
    payload = {
        "schema": "sparkinfer.w4a8.fc1_raw_diagnostic.v1",
        "toolchain": {
            "cutlass_dsl": version,
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            [
                {
                    key: case[key]
                    for key in (
                        "tile_m",
                        "rows",
                        "cos_mean",
                        "cos_min",
                        "rmse_mean",
                        "rmse_max",
                        "max_abs",
                        "bad_row_count",
                        "bands",
                    )
                }
                for case in cases
            ],
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
