#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def _write_text(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _ensure_package_dir(path: pathlib.Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    init_path = path / "__init__.py"
    if not init_path.exists():
        _write_text(init_path, "")


def _load_points(
    *,
    checkpoint_jsonl: list[pathlib.Path],
    input_json: list[pathlib.Path],
) -> dict[int, dict[str, object]]:
    points: dict[int, dict[str, object]] = {}
    for checkpoint_path in checkpoint_jsonl:
        with checkpoint_path.open(encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                if payload.get("type") != "point_complete":
                    continue
                point = payload["payload"]
                points[int(point["routed_rows"])] = point
    for input_path in input_json:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        for point in payload.get("points", []):
            points[int(point["routed_rows"])] = point
    return points


def _preferred_mac(point: dict[str, object], backend: str) -> int | None:
    preferred = point["backend_results"][backend]["preferred_winner"]
    if preferred is None:
        return None
    return int(preferred["requested_max_active_clusters"])


def _backend_ladder(
    *,
    points: dict[int, dict[str, object]],
    backend: str,
    include_predicate,
) -> tuple[tuple[int, int], ...]:
    ladder: list[tuple[int, int]] = []
    for routed_rows in sorted(points):
        if not include_predicate(int(routed_rows)):
            continue
        preferred_mac = _preferred_mac(points[int(routed_rows)], backend)
        if preferred_mac is None:
            continue
        ladder.append((int(routed_rows), int(preferred_mac)))
    if not ladder:
        raise ValueError(f"no preferred winners found for backend={backend!r}")
    return tuple(ladder)


def _format_ladder(ladder: tuple[tuple[int, int], ...]) -> str:
    return "\n".join(
        f"        ({int(end_routed_rows)}, {int(max_active_clusters)}),"
        for end_routed_rows, max_active_clusters in ladder
    )


def _module_text(
    *,
    regime: str,
    micro_ladder: tuple[tuple[int, int], ...],
    static_ladder: tuple[tuple[int, int], ...],
    dynamic_ladder: tuple[tuple[int, int], ...],
    micro_cutover_rows: int,
    static_cutover_rows: int,
) -> str:
    return (
        '"""Generated MoE decode MAX_ACTIVE_CLUSTERS tuning data."""\n\n'
        "from .registry import register_max_active_clusters_policy\n\n"
        f"# micro: routed_rows <= {int(micro_cutover_rows)}\n"
        f"# static: {int(micro_cutover_rows)} < routed_rows <= {int(static_cutover_rows)}\n"
        f"# dynamic: routed_rows > {int(static_cutover_rows)}\n\n"
        "register_max_active_clusters_policy(\n"
        f'    regime="{regime}",\n'
        '    backend="micro",\n'
        "    ladder=(\n"
        f"{_format_ladder(micro_ladder)}\n"
        "    ),\n"
        ")\n\n"
        "register_max_active_clusters_policy(\n"
        f'    regime="{regime}",\n'
        '    backend="static",\n'
        "    ladder=(\n"
        f"{_format_ladder(static_ladder)}\n"
        "    ),\n"
        ")\n\n"
        "register_max_active_clusters_policy(\n"
        f'    regime="{regime}",\n'
        '    backend="dynamic",\n'
        "    ladder=(\n"
        f"{_format_ladder(dynamic_ladder)}\n"
        "    ),\n"
        ")\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-jsonl", action="append", default=[])
    parser.add_argument("--input-json", action="append", default=[])
    parser.add_argument(
        "--output-root",
        default=str(pathlib.Path(__file__).resolve().parents[1] / "sparkinfer" / "moe" / "tuning"),
    )
    parser.add_argument("--output-name", default="decode.max_active_clusters.py")
    parser.add_argument("--regime", default="decode")
    parser.add_argument("--micro-cutover-rows", type=int, default=20)
    parser.add_argument("--static-cutover-rows", type=int, default=640)
    args = parser.parse_args()

    checkpoint_paths = [pathlib.Path(value) for value in args.checkpoint_jsonl]
    input_paths = [pathlib.Path(value) for value in args.input_json]
    if not checkpoint_paths and not input_paths:
        raise ValueError("expected at least one --checkpoint-jsonl or --input-json")

    points = _load_points(
        checkpoint_jsonl=checkpoint_paths,
        input_json=input_paths,
    )
    if not points:
        raise ValueError("no point payloads found in inputs")

    micro_ladder = _backend_ladder(
        points=points,
        backend="micro",
        include_predicate=lambda routed_rows: routed_rows <= int(args.micro_cutover_rows),
    )
    static_ladder = _backend_ladder(
        points=points,
        backend="static",
        include_predicate=lambda routed_rows: int(args.micro_cutover_rows) < routed_rows <= int(args.static_cutover_rows),
    )
    dynamic_ladder = _backend_ladder(
        points=points,
        backend="dynamic",
        include_predicate=lambda routed_rows: routed_rows > int(args.static_cutover_rows),
    )

    output_root = pathlib.Path(args.output_root)
    _ensure_package_dir(output_root)
    output_path = output_root / args.output_name
    _write_text(
        output_path,
        _module_text(
            regime=str(args.regime),
            micro_ladder=micro_ladder,
            static_ladder=static_ladder,
            dynamic_ladder=dynamic_ladder,
            micro_cutover_rows=int(args.micro_cutover_rows),
            static_cutover_rows=int(args.static_cutover_rows),
        ),
    )
    print(output_path)


if __name__ == "__main__":
    main()
