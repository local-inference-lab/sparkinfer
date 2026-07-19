"""Capture and compare W4A8 FC1 activation-quantization intermediates on GPU."""

from __future__ import annotations

import argparse
import hashlib
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

    from b12x.cute.runtime_patches import apply_cutlass_runtime_patches

    apply_cutlass_runtime_patches()
    return version


def _sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def _cosine_rows(got: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    got = got.float()
    ref = ref.float()
    dot = (got * ref).sum(dim=1)
    got_norm = got.norm(dim=1)
    ref_norm = ref.norm(dim=1)
    denom = got_norm * ref_norm
    both_zero = (got_norm == 0) & (ref_norm == 0)
    return torch.where(
        both_zero,
        torch.ones_like(dot),
        torch.where(denom > 0, dot / denom, torch.zeros_like(dot)),
    )


def _route_slot(topk_ids: torch.Tensor, token: int, expert: int) -> int:
    matches = torch.nonzero(topk_ids[token] == expert, as_tuple=False).flatten()
    if matches.numel() != 1:
        raise RuntimeError(
            f"expected one route for token={token}, expert={expert}; got {matches.tolist()}"
        )
    return int(matches[0].item())


def _analyze_case(
    run_w4a8_dynamic: Any,
    *,
    tile_m: int,
    m: int,
    route_weight_slot: int | None,
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
        route_weight_slot=route_weight_slot,
        return_debug=True,
        capture_intermediate=True,
    )
    topk_ids = debug["topk_ids"]
    token_map = debug["token_map"]
    expert = debug["intermediate_physical_expert"]
    expert_local = debug["intermediate_physical_expert_local"]
    valid = expert >= 0
    physical_rows = torch.nonzero(valid, as_tuple=False).flatten()

    device_payload = debug["intermediate_device_payload"][valid]
    reference_payload = debug["intermediate_reference_payload"][valid]
    device_scale = debug["intermediate_device_scale"][valid]
    reference_scale = debug["intermediate_reference_scale"][valid]
    device_dequant = debug["intermediate_device_dequant"][valid]
    reference_dequant = debug["intermediate_reference_dequant"][valid]
    reference_activated = debug["intermediate_reference_activated"][valid]

    row_cos = _cosine_rows(device_dequant, reference_dequant)
    row_error = device_dequant.float() - reference_dequant.float()
    row_rmse = row_error.square().mean(dim=1).sqrt()
    payload_exact = (device_payload == reference_payload).float().mean(dim=1)
    scale_exact = (device_scale == reference_scale).float().mean(dim=1)
    device_zero_blocks = device_dequant.view(-1, 4, 32).eq(0).all(dim=2)
    reference_zero_blocks = reference_activated.view(-1, 4, 32).eq(0).all(dim=2)
    zero_mismatch = device_zero_blocks != reference_zero_blocks

    per_row = []
    for packed_idx, physical_row_tensor in enumerate(physical_rows):
        physical_row = int(physical_row_tensor.item())
        token = int(token_map[physical_row].item())
        eid = int(expert[physical_row].item())
        local = int(expert_local[physical_row].item())
        per_row.append(
            {
                "physical_row": physical_row,
                "token": token,
                "expert": eid,
                "expert_local": local,
                "expert_local_mod_tile": local % tile_m,
                "route_slot": _route_slot(topk_ids, token, eid),
                "cos": float(row_cos[packed_idx].item()),
                "rmse": float(row_rmse[packed_idx].item()),
                "payload_exact_fraction": float(payload_exact[packed_idx].item()),
                "scale_exact_fraction": float(scale_exact[packed_idx].item()),
                "reference_zero_blocks": reference_zero_blocks[packed_idx].tolist(),
                "device_zero_blocks": device_zero_blocks[packed_idx].tolist(),
                "zero_block_mismatch": zero_mismatch[packed_idx].tolist(),
                "reference_scale_bytes": reference_scale[packed_idx].tolist(),
                "device_scale_bytes": device_scale[packed_idx].tolist(),
                "reference_payload_hex": bytes(reference_payload[packed_idx].tolist()).hex(),
                "device_payload_hex": bytes(device_payload[packed_idx].tolist()).hex(),
            }
        )

    bad_rows = [row for row in per_row if row["cos"] < 0.999]
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
        idx_tensor = torch.tensor(indices, dtype=torch.long)
        bands.append(
            {
                "name": name,
                "rows": len(indices),
                "cos_mean": float(row_cos[idx_tensor].mean().item()),
                "cos_min": float(row_cos[idx_tensor].min().item()),
                "rmse_mean": float(row_rmse[idx_tensor].mean().item()),
                "payload_exact_fraction": float(
                    payload_exact[idx_tensor].mean().item()
                ),
                "scale_exact_fraction": float(scale_exact[idx_tensor].mean().item()),
                "zero_block_mismatch_count": int(
                    zero_mismatch[idx_tensor].sum().item()
                ),
            }
        )

    output_cos = _cosine_rows(output.detach().cpu(), reference.detach().cpu())
    return {
        "tile_m": tile_m,
        "m": m,
        "route_weight_slot": route_weight_slot,
        "output": {
            "cos_mean": float(output_cos.mean().item()),
            "cos_min": float(output_cos.min().item()),
            "bad_rows": torch.nonzero(output_cos < 0.999, as_tuple=False)
            .flatten()
            .tolist(),
        },
        "intermediate": {
            "rows": int(valid.sum().item()),
            "cos_mean": float(row_cos.mean().item()),
            "cos_min": float(row_cos.min().item()),
            "rmse_mean": float(row_rmse.mean().item()),
            "rmse_max": float(row_rmse.max().item()),
            "payload_exact_fraction": float(payload_exact.mean().item()),
            "scale_exact_fraction": float(scale_exact.mean().item()),
            "reference_zero_block_count": int(reference_zero_blocks.sum().item()),
            "device_zero_block_count": int(device_zero_blocks.sum().item()),
            "zero_block_mismatch_count": int(zero_mismatch.sum().item()),
            "device_payload_sha256": _sha256(device_payload),
            "reference_payload_sha256": _sha256(reference_payload),
            "device_scale_sha256": _sha256(device_scale),
            "reference_scale_sha256": _sha256(reference_scale),
            "bad_row_count": len(bad_rows),
            "bad_rows": bad_rows,
            "bands": bands,
            "per_row": per_row,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_target_gpu_argument(parser)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tiles", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument(
        "--route-slots", type=int, nargs="+", default=[-1, 0, 1],
        help="-1 keeps both routes; 0/1 isolates that route's output weight",
    )
    args = parser.parse_args()

    gpu_scope = require_target_gpu(args.expected_physical_gpu)
    version = _prepare_cutlass_runtime()

    from tests.test_w4a8_dynamic_kernel import _run_w4a8_dynamic

    cases = []
    for tile_m in args.tiles:
        m = {16: 16, 32: 33, 64: 65}[tile_m]
        for slot_arg in args.route_slots:
            slot = None if slot_arg < 0 else slot_arg
            cases.append(
                _analyze_case(
                    _run_w4a8_dynamic,
                    tile_m=tile_m,
                    m=m,
                    route_weight_slot=slot,
                )
            )

    payload = {
        "schema": "b12x.w4a8.fc1_intermediate_diagnostic.v1",
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
    summary = [
        {
            "tile_m": case["tile_m"],
            "slot": case["route_weight_slot"],
            "output_bad": case["output"]["bad_rows"],
            "intermediate_bad_count": case["intermediate"]["bad_row_count"],
            "intermediate_cos_min": case["intermediate"]["cos_min"],
            "payload_exact_fraction": case["intermediate"][
                "payload_exact_fraction"
            ],
            "scale_exact_fraction": case["intermediate"]["scale_exact_fraction"],
            "zero_block_mismatch_count": case["intermediate"][
                "zero_block_mismatch_count"
            ],
        }
        for case in cases
    ]
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
