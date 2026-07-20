#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def _normalize_kv_dtype_key(kv_dtype: str) -> str:
    return {
        "bf16": "bf16",
        "bfloat16": "bf16",
        "fp16": "fp16",
        "float16": "fp16",
        "fp8": "fp8",
        "fp8_e4m3fn": "fp8",
        "float8_e4m3fn": "fp8",
    }.get(kv_dtype, kv_dtype)


def _write_text(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _ensure_package_dir(path: pathlib.Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    init_path = path / "__init__.py"
    if not init_path.exists():
        _write_text(init_path, "")


def _format_ladder(ladder: list[dict[str, int]]) -> str:
    rows = [
        f"        ({int(segment['end_page'])}, {int(segment['winner_fixed_split_pages'])}),"
        for segment in ladder
    ]
    return "\n".join(rows)


def _collapse_monotone_chunk_ladder(
    *,
    page_payloads: list[dict[str, object]],
    page_size: int,
) -> list[dict[str, int]]:
    page_rows: list[tuple[int, list[int]]] = []
    for page_payload in sorted(page_payloads, key=lambda row: int(row["page_count"])):
        cta_results = page_payload.get("cta_results")
        if not isinstance(cta_results, list) or len(cta_results) != 1:
            continue
        cta_result = cta_results[0]
        tied_chunk_winners = cta_result.get("tied_chunk_winners")
        if not isinstance(tied_chunk_winners, list) or not tied_chunk_winners:
            continue
        winners = sorted({int(summary["fixed_split_pages"]) for summary in tied_chunk_winners})
        if not winners:
            continue
        page_rows.append((int(page_payload["page_count"]), winners))

    if not page_rows:
        return []

    collapsed: list[dict[str, int]] = []
    current_start = page_rows[0][0]
    current_end = page_rows[0][0]
    current_winner = int(page_rows[0][1][0])

    for page_count, winners in page_rows[1:]:
        if current_winner in winners:
            chosen = int(current_winner)
        else:
            nondecreasing = [int(winner) for winner in winners if int(winner) >= int(current_winner)]
            chosen = min(nondecreasing) if nondecreasing else min(winners)

        if chosen < current_winner:
            chosen = int(current_winner)

        if chosen == current_winner and page_count == current_end + 1:
            current_end = int(page_count)
            continue

        collapsed.append(
            {
                "start_page": int(current_start),
                "end_page": int(current_end),
                "start_cache_tokens": int(current_start * page_size),
                "end_cache_tokens": int(current_end * page_size),
                "winner_fixed_split_pages": int(current_winner),
            }
        )
        current_start = int(page_count)
        current_end = int(page_count)
        current_winner = int(chosen)

    collapsed.append(
        {
            "start_page": int(current_start),
            "end_page": int(current_end),
            "start_cache_tokens": int(current_start * page_size),
            "end_cache_tokens": int(current_end * page_size),
            "winner_fixed_split_pages": int(current_winner),
        }
    )
    return collapsed


def _module_text(
    *,
    kv_dtype: str,
    regime: str,
    batch: int,
    graph_ctas_per_sm: int,
    capture_fixed_split_pages: int | None,
    capture_page_count: int,
    page_size: int,
    chunk_ladder: list[dict[str, int]],
) -> str:
    capture_fixed_literal = "None" if capture_fixed_split_pages is None else str(int(capture_fixed_split_pages))
    return (
        '"""Generated decode graph policy tuning data."""\n\n'
        "from .registry import register_decode_graph_policy\n\n"
        "register_decode_graph_policy(\n"
        f'    kv_dtype="{kv_dtype}",\n'
        f'    regime="{regime}",\n'
        f"    batch={int(batch)},\n"
        f"    graph_ctas_per_sm={int(graph_ctas_per_sm)},\n"
        f"    capture_fixed_split_pages={capture_fixed_literal},\n"
        f"    capture_page_count={int(capture_page_count)},\n"
        f"    page_size={int(page_size)},\n"
        "    chunk_ladder=(\n"
        f"{_format_ladder(chunk_ladder)}\n"
        "    ),\n"
        ")\n"
    )


def _batch_payload_to_module(
    *,
    payload: dict[str, object],
    kv_dtype: str,
    regime: str,
    capture_page_count: int,
    page_size: int,
) -> tuple[int, str]:
    batch = int(payload["batch"])
    chunk_fill = payload["chunk_fill"]
    graph_ctas_per_sm = int(chunk_fill["graph_ctas_per_sm"])
    capture_fixed_split_pages = chunk_fill["capture_fixed_split_pages"]
    chunk_ladder = _collapse_monotone_chunk_ladder(
        page_payloads=list(chunk_fill["pages"]),
        page_size=page_size,
    )
    if not chunk_ladder:
        chunk_ladder = chunk_fill["chunk_ladder"]
    if not isinstance(chunk_ladder, list) or not chunk_ladder:
        raise ValueError(f"expected non-empty chunk_fill.chunk_ladder for batch={batch}")

    return (
        batch,
        _module_text(
            kv_dtype=kv_dtype,
            regime=regime,
            batch=batch,
            graph_ctas_per_sm=graph_ctas_per_sm,
            capture_fixed_split_pages=(
                None if capture_fixed_split_pages is None else int(capture_fixed_split_pages)
            ),
            capture_page_count=capture_page_count,
            page_size=page_size,
            chunk_ladder=chunk_ladder,
        ),
    )


def _generate_from_input(
    *,
    input_path: pathlib.Path,
    output_root: pathlib.Path,
    selected_batches: set[int] | None,
) -> list[pathlib.Path]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    config = payload["config"]
    regime = str(config["mode"])
    if regime != "decode":
        raise ValueError(f"only decode policy generation is supported for now, got mode={regime!r}")
    kv_dtype = _normalize_kv_dtype_key(str(config["kv_dtype"]))
    capture_page_count = int(config["capture_page_count"])
    page_size = int(config["page_size"])

    _ensure_package_dir(output_root)

    written: list[pathlib.Path] = []
    for batch_payload in payload["batches"]:
        batch = int(batch_payload["batch"])
        if selected_batches is not None and batch not in selected_batches:
            continue
        batch_value, module_text = _batch_payload_to_module(
            payload=batch_payload,
            kv_dtype=kv_dtype,
            regime=regime,
            capture_page_count=capture_page_count,
            page_size=page_size,
        )
        output_path = output_root / f"{kv_dtype}.{regime}.bs{batch_value}.py"
        _write_text(output_path, module_text)
        written.append(output_path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True, help="Combined decode-policy JSON input")
    parser.add_argument(
        "--output-root",
        default=str(pathlib.Path(__file__).resolve().parents[1] / "sparkinfer" / "attention" / "paged" / "tuning"),
    )
    parser.add_argument(
        "--batch-list",
        type=str,
        default="",
        help="Optional comma-separated batch subset to emit from each input",
    )
    args = parser.parse_args()

    selected_batches = None
    if args.batch_list:
        selected_batches = {int(part) for part in args.batch_list.split(",") if part.strip()}
        if not selected_batches:
            raise ValueError("expected positive batch sizes in --batch-list")

    output_root = pathlib.Path(args.output_root)
    written_paths: list[pathlib.Path] = []
    for raw_input in args.input:
        written_paths.extend(
            _generate_from_input(
                input_path=pathlib.Path(raw_input),
                output_root=output_root,
                selected_batches=selected_batches,
            )
        )

    for path in written_paths:
        print(path)


if __name__ == "__main__":
    main()
