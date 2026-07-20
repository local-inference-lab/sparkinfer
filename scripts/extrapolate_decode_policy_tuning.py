#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import pathlib
import re
import sys
from dataclasses import dataclass

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


@dataclass(frozen=True)
class DecodePolicy:
    kv_dtype: str
    regime: str
    batch: int
    graph_ctas_per_sm: int
    capture_fixed_split_pages: int | None
    capture_page_count: int
    page_size: int
    chunk_ladder: tuple[tuple[int, int], ...]


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


def _parse_csv_ints(raw: str) -> list[int]:
    values = [int(part) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("expected at least one integer value")
    return sorted({value for value in values if value > 0})


def _parse_csv_strs(raw: str) -> list[str]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("expected at least one string value")
    return values


def _load_policy_module(path: pathlib.Path) -> DecodePolicy:
    captured: dict[str, object] = {}

    def register_decode_graph_policy(**kwargs: object) -> None:
        captured.update(kwargs)

    namespace = {"register_decode_graph_policy": register_decode_graph_policy}
    source = "\n".join(
        line for line in path.read_text(encoding="utf-8").splitlines() if "from .registry import" not in line
    )
    exec(source, namespace, namespace)
    chunk_ladder = tuple((int(end_page), int(chunk_pages)) for end_page, chunk_pages in captured["chunk_ladder"])
    return DecodePolicy(
        kv_dtype=_normalize_kv_dtype_key(str(captured["kv_dtype"])),
        regime=str(captured["regime"]),
        batch=int(captured["batch"]),
        graph_ctas_per_sm=int(captured["graph_ctas_per_sm"]),
        capture_fixed_split_pages=(
            None if captured["capture_fixed_split_pages"] is None else int(captured["capture_fixed_split_pages"])
        ),
        capture_page_count=int(captured["capture_page_count"]),
        page_size=int(captured["page_size"]),
        chunk_ladder=chunk_ladder,
    )


def _load_policies(input_root: pathlib.Path, regime: str) -> dict[str, dict[int, DecodePolicy]]:
    policies: dict[str, dict[int, DecodePolicy]] = {}
    pattern = re.compile(r"(bf16|fp8)\." + re.escape(regime) + r"\.bs(\d+)\.py$")
    for path in sorted(input_root.glob(f"*.{regime}.bs*.py")):
        match = pattern.match(path.name)
        if not match:
            continue
        policy = _load_policy_module(path)
        policies.setdefault(policy.kv_dtype, {})[policy.batch] = policy
    return policies


def _collapse_ladder(rows: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    if not rows:
        return ()
    collapsed: list[tuple[int, int]] = []
    previous_chunk: int | None = None
    for end_page, chunk_pages in rows:
        if previous_chunk == chunk_pages:
            continue
        collapsed.append((int(end_page), int(chunk_pages)))
        previous_chunk = int(chunk_pages)
    if collapsed[-1][0] != rows[-1][0]:
        collapsed.append((int(rows[-1][0]), int(rows[-1][1])))
    elif collapsed[-1][1] != rows[-1][1]:
        collapsed.append((int(rows[-1][0]), int(rows[-1][1])))
    return tuple(collapsed)


def _scale_ladder(source: DecodePolicy, target_batch: int) -> tuple[tuple[int, int], ...]:
    scale = float(target_batch) / float(source.batch)
    scaled_rows: list[tuple[int, int]] = []
    previous_chunk = 1
    for end_page, source_chunk in source.chunk_ladder:
        scaled_chunk = int(round(float(source_chunk) * scale))
        scaled_chunk = max(1, min(int(end_page), int(source.capture_page_count), scaled_chunk))
        if scaled_chunk < previous_chunk:
            scaled_chunk = int(previous_chunk)
        scaled_rows.append((int(end_page), int(scaled_chunk)))
        previous_chunk = int(scaled_chunk)
    return _collapse_ladder(scaled_rows)


def _scale_capture_fixed_split_pages(source: DecodePolicy, target_batch: int) -> int | None:
    if source.capture_fixed_split_pages is None:
        return None
    scaled = int(math.ceil(float(source.capture_fixed_split_pages) * float(target_batch) / float(source.batch)))
    return max(1, min(int(source.capture_page_count), scaled))


def _module_text(*, source: DecodePolicy, target_batch: int, chunk_ladder: tuple[tuple[int, int], ...]) -> str:
    capture_fixed_split_pages = _scale_capture_fixed_split_pages(source, target_batch)
    capture_literal = "None" if capture_fixed_split_pages is None else str(int(capture_fixed_split_pages))
    ladder_rows = "\n".join(f"        ({end_page}, {chunk_pages})," for end_page, chunk_pages in chunk_ladder)
    return (
        '"""Synthetic decode graph policy extrapolated from an existing higher-batch seed."""\n\n'
        "from .registry import register_decode_graph_policy\n\n"
        f"# source_batch={source.batch}\n"
        "register_decode_graph_policy(\n"
        f'    kv_dtype="{source.kv_dtype}",\n'
        f'    regime="{source.regime}",\n'
        f"    batch={int(target_batch)},\n"
        f"    graph_ctas_per_sm={int(source.graph_ctas_per_sm)},\n"
        f"    capture_fixed_split_pages={capture_literal},\n"
        f"    capture_page_count={int(source.capture_page_count)},\n"
        f"    page_size={int(source.page_size)},\n"
        "    chunk_ladder=(\n"
        f"{ladder_rows}\n"
        "    ),\n"
        ")\n"
    )


def _write_text(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        default=str(pathlib.Path(__file__).resolve().parents[1] / "sparkinfer" / "attention" / "paged" / "tuning"),
    )
    parser.add_argument(
        "--output-root",
        default=str(pathlib.Path(__file__).resolve().parents[1] / "sparkinfer" / "attention" / "paged" / "tuning"),
    )
    parser.add_argument("--dtype-list", default="bf16,fp8")
    parser.add_argument("--batch-list", default="32,64,128")
    parser.add_argument("--regime", default="decode")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    input_root = pathlib.Path(args.input_root).resolve()
    output_root = pathlib.Path(args.output_root).resolve()
    dtypes = [_normalize_kv_dtype_key(value) for value in _parse_csv_strs(args.dtype_list)]
    target_batches = _parse_csv_ints(args.batch_list)
    policies = _load_policies(input_root, args.regime)

    written: list[pathlib.Path] = []
    for kv_dtype in dtypes:
        family = policies.get(kv_dtype, {})
        if not family:
            raise ValueError(f"no source policies found for dtype={kv_dtype!r} in {input_root}")
        available_batches = sorted(family)
        source_batch = available_batches[-1]
        source = family[source_batch]
        for target_batch in target_batches:
            if target_batch in family:
                source = family[target_batch]
                source_batch = int(target_batch)
                continue
            chunk_ladder = _scale_ladder(source, target_batch)
            module_text = _module_text(
                source=source,
                target_batch=target_batch,
                chunk_ladder=chunk_ladder,
            )
            output_path = output_root / f"{kv_dtype}.{args.regime}.bs{target_batch}.py"
            _write_text(output_path, module_text)
            written.append(output_path)
            if args.summary:
                print(
                    f"# wrote {output_path} source_batch={source_batch} "
                    f"cta={source.graph_ctas_per_sm} capture={_scale_capture_fixed_split_pages(source, target_batch)}"
                )

    if args.summary and not written:
        print("# no files written")


if __name__ == "__main__":
    main()
