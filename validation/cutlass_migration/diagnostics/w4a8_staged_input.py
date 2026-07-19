"""Compare staged W4A8 FC1 A/SFA bytes with their routed global source."""

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
    from b12x.cute.runtime_patches import apply_cutlass_runtime_patches

    apply_cutlass_runtime_patches()
    return version


def _route_slot(topk_ids: torch.Tensor, token: int, expert: int) -> int:
    match = torch.nonzero(topk_ids[token] == expert, as_tuple=False).flatten()
    if match.numel() != 1:
        raise RuntimeError(
            f"expected one route for token={token}, expert={expert}; "
            f"got {match.tolist()}"
        )
    return int(match[0].item())


def _packet_comparison(
    device_payload: torch.Tensor,
    source_payload: torch.Tensor,
    device_scale: torch.Tensor,
    source_scale: torch.Tensor,
    packet: int,
) -> dict[str, Any]:
    payload_begin = packet * 128
    scale_begin = packet * 4
    payload_equal = (
        device_payload[payload_begin : payload_begin + 128]
        == source_payload[payload_begin : payload_begin + 128]
    )
    scale_equal = (
        device_scale[scale_begin : scale_begin + 4]
        == source_scale[scale_begin : scale_begin + 4]
    )
    payload_mismatch = torch.nonzero(~payload_equal, as_tuple=False).flatten()
    return {
        "packet": packet,
        "payload_exact": bool(payload_equal.all().item()),
        "payload_exact_fraction": float(payload_equal.float().mean().item()),
        "payload_mismatch_indices": payload_mismatch.tolist(),
        "payload_device_at_mismatch": device_payload[
            payload_begin + payload_mismatch
        ].tolist(),
        "payload_source_at_mismatch": source_payload[
            payload_begin + payload_mismatch
        ].tolist(),
        "scale_exact": bool(scale_equal.all().item()),
        "scale_device": device_scale[scale_begin : scale_begin + 4].tolist(),
        "scale_source": source_scale[scale_begin : scale_begin + 4].tolist(),
    }


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
        capture_fc1_staged=True,
    )

    expert = debug["fc1_staged_physical_expert"]
    expert_local = debug["fc1_staged_physical_expert_local"]
    valid = expert >= 0
    physical_rows = torch.nonzero(valid, as_tuple=False).flatten()
    device_payload = debug["fc1_staged_device_payload"][valid]
    source_payload = debug["fc1_staged_source_payload"][valid]
    device_scale = debug["fc1_staged_device_scale"][valid]
    source_scale = debug["fc1_staged_source_scale"][valid]
    source_blocks = source_payload.view(source_payload.shape[0], 2, 128)
    down_residual_blocks = debug["fc1_staged_down_residual"].view(4, -1, 128)

    packet_summaries = []
    for packet in range(2):
        payload_equal = (
            device_payload[:, packet * 128 : (packet + 1) * 128]
            == source_payload[:, packet * 128 : (packet + 1) * 128]
        )
        scale_equal = (
            device_scale[:, packet * 4 : (packet + 1) * 4]
            == source_scale[:, packet * 4 : (packet + 1) * 4]
        )
        packet_summaries.append(
            {
                "packet": packet,
                "rows": int(physical_rows.numel()),
                "payload_exact_rows": int(payload_equal.all(dim=1).sum().item()),
                "payload_mismatch_bytes": int((~payload_equal).sum().item()),
                "scale_exact_rows": int(scale_equal.all(dim=1).sum().item()),
                "scale_mismatch_bytes": int((~scale_equal).sum().item()),
            }
        )

    target_rows = []
    for idx, physical_row_tensor in enumerate(physical_rows):
        physical_row = int(physical_row_tensor.item())
        local = int(expert_local[physical_row].item())
        if local not in (15, 16, 17):
            continue
        eid = int(expert[physical_row].item())
        token = int(debug["token_map"][physical_row].item())
        packets = []
        for packet in range(2):
            comparison = _packet_comparison(
                device_payload[idx],
                source_payload[idx],
                device_scale[idx],
                source_scale[idx],
                packet,
            )
            staged = device_payload[idx, packet * 128 : (packet + 1) * 128]
            source_matches = torch.nonzero(
                (source_blocks == staged).all(dim=2), as_tuple=False
            )
            comparison["activation_source_matches"] = [
                {
                    "physical_row": int(physical_rows[source_row].item()),
                    "packet": int(source_packet),
                }
                for source_row, source_packet in source_matches.tolist()
            ]
            residual_matches = torch.nonzero(
                (down_residual_blocks == staged).all(dim=2), as_tuple=False
            )
            comparison["down_residual_matches"] = [
                {"expert": int(match_expert), "block": int(block)}
                for match_expert, block in residual_matches.tolist()
            ]
            packets.append(comparison)
        target_rows.append(
            {
                "physical_row": physical_row,
                "token": token,
                "expert": eid,
                "expert_local": local,
                "route_slot": _route_slot(debug["topk_ids"], token, eid),
                "packets": packets,
            }
        )

    output_f = output.float()
    reference_f = reference.float()
    output_cos = float(
        torch.nn.functional.cosine_similarity(
            output_f.flatten(), reference_f.flatten(), dim=0
        ).item()
    )
    output_rmse = float((output_f - reference_f).square().mean().sqrt().item())
    return {
        "tile_m": tile_m,
        "m": m,
        "valid_rows": int(physical_rows.numel()),
        "output_cos": output_cos,
        "output_rmse": output_rmse,
        "packets": packet_summaries,
        "target_rows": target_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_target_gpu_argument(parser)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tiles", type=int, nargs="+", default=[32])
    args = parser.parse_args()

    gpu_scope = require_target_gpu(args.expected_physical_gpu)
    version = _prepare_cutlass_runtime()
    from tests.test_w4a8_dynamic_kernel import _run_w4a8_dynamic

    cases = [_case(_run_w4a8_dynamic, tile_m) for tile_m in args.tiles]
    payload = {
        "schema": "b12x.w4a8.staged_input_diagnostic.v1",
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
                        "valid_rows",
                        "output_cos",
                        "output_rmse",
                        "packets",
                    )
                }
                | {
                    "targets": [
                        {
                            "expert": row["expert"],
                            "expert_local": row["expert_local"],
                            "packet_payload_exact": [
                                packet["payload_exact"]
                                for packet in row["packets"]
                            ],
                            "packet_down_residual_matches": [
                                packet["down_residual_matches"]
                                for packet in row["packets"]
                            ],
                        }
                        for row in case["target_rows"]
                    ]
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
