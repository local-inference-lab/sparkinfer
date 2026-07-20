#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from sparkinfer.attention._shared.contiguous.api import clear_attention_caches
from sparkinfer.attention.paged.workspace import PagedAttentionWorkspace
from tests.test_attention_paged_planner import _make_inputs

_LANES = 32
_HEAD_DIM = 256
_WORDS_PER_LANE = (_HEAD_DIM // 16) * 4
_DUMP_WORDS = _LANES * _WORDS_PER_LANE
_SEED = 1234


@contextlib.contextmanager
def _env(overrides: dict[str, str | None]):
    previous = {key: os.environ.get(key) for key in overrides}
    try:
        for key, value in overrides.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _dump_words(
    *,
    use_tma: bool,
    consumer: str,
    plain_bf16_layout: bool = False,
) -> torch.Tensor:
    if consumer not in {"manual", "donor"}:
        raise ValueError(f"unsupported consumer={consumer}")
    overrides = {
        "SPARKINFER_PAGED_KV_DEBUG_DUMP": "PVREGS" if consumer == "manual" else "PVREGS_DONOR",
        "SPARKINFER_PAGED_KV_TMA": "1" if use_tma else None,
        "SPARKINFER_PAGED_KV_TMA_PLAIN_BF16_LAYOUT": "1" if (use_tma and plain_bf16_layout) else None,
        "SPARKINFER_PAGED_KV_TMA_COPYFRAG_QK": None,
        "SPARKINFER_PAGED_KV_TMA_COPYFRAG_PV": None,
        "SPARKINFER_PAGED_KV_TMA_DONOR_GEMM": None,
    }
    with _env(overrides):
        clear_attention_caches()
        torch.manual_seed(_SEED)
        q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
            q_seqlens=[1, 1, 1],
            cache_seqlens=[64, 128, 192],
            dtype=torch.bfloat16,
            kv_dtype=torch.bfloat16,
        )
        workspace = PagedAttentionWorkspace.for_tensors(
            mode="decode",
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
        )
        workspace.prepare(
            page_table,
            cache_seqlens,
            cu_seqlens_q,
            disable_split_kv=True,
        )
        output = torch.zeros_like(q)
        workspace.run(q, k_cache, v_cache, output=output)
        torch.cuda.synchronize()
    raw8 = output.view(torch.uint8).reshape(-1).to(torch.int64)
    words = (
        raw8[0::4]
        | (raw8[1::4] << 8)
        | (raw8[2::4] << 16)
        | (raw8[3::4] << 24)
    )
    return words[:_DUMP_WORDS].cpu()


def _lane_words(words: torch.Tensor, lane: int) -> list[int]:
    start = lane * _WORDS_PER_LANE
    end = start + _WORDS_PER_LANE
    return [int(v) for v in words[start:end].tolist()]


def _hex_words(words: list[int], count: int) -> list[str]:
    return [f"0x{value & 0xFFFFFFFF:08x}" for value in words[:count]]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare raw BF16 PV operand registers across paged-kernel backend/consumer combinations."
    )
    parser.add_argument(
        "--consumer",
        choices=("manual", "donor"),
        default=None,
        help="consumer path for both sides; shorthand for --lhs-consumer/--rhs-consumer",
    )
    parser.add_argument(
        "--lhs-backend",
        choices=("cpasync", "tma"),
        default="cpasync",
        help="backend for the left-hand dump",
    )
    parser.add_argument(
        "--rhs-backend",
        choices=("cpasync", "tma"),
        default="tma",
        help="backend for the right-hand dump",
    )
    parser.add_argument(
        "--lhs-consumer",
        choices=("manual", "donor"),
        default=None,
        help="consumer path for the left-hand dump",
    )
    parser.add_argument(
        "--rhs-consumer",
        choices=("manual", "donor"),
        default=None,
        help="consumer path for the right-hand dump",
    )
    parser.add_argument(
        "--lhs-plain-bf16-layout",
        action="store_true",
        help="enable SPARKINFER_PAGED_KV_TMA_PLAIN_BF16_LAYOUT for the left-hand TMA dump",
    )
    parser.add_argument(
        "--rhs-plain-bf16-layout",
        action="store_true",
        help="enable SPARKINFER_PAGED_KV_TMA_PLAIN_BF16_LAYOUT for the right-hand TMA dump",
    )
    parser.add_argument(
        "--show-lane",
        type=int,
        default=0,
        help="lane to include in the detailed hex dump",
    )
    parser.add_argument(
        "--show-words",
        type=int,
        default=16,
        help="number of words from the selected lane to print",
    )
    parser.add_argument(
        "--json-out",
        type=pathlib.Path,
        default=None,
        help="optional path to write the full JSON report",
    )
    args = parser.parse_args()

    if args.show_lane < 0 or args.show_lane >= _LANES:
        raise ValueError(f"--show-lane must be in [0, {_LANES - 1}]")

    lhs_consumer = args.consumer if args.consumer is not None else args.lhs_consumer
    rhs_consumer = args.consumer if args.consumer is not None else args.rhs_consumer
    if lhs_consumer is None or rhs_consumer is None:
        raise ValueError("specify --consumer or both --lhs-consumer and --rhs-consumer")

    lhs = _dump_words(
        use_tma=args.lhs_backend == "tma",
        consumer=lhs_consumer,
        plain_bf16_layout=args.lhs_plain_bf16_layout,
    )
    rhs = _dump_words(
        use_tma=args.rhs_backend == "tma",
        consumer=rhs_consumer,
        plain_bf16_layout=args.rhs_plain_bf16_layout,
    )
    mismatches = (lhs != rhs).nonzero(as_tuple=False).reshape(-1)
    mismatch_lanes = sorted({int(idx.item()) // _WORDS_PER_LANE for idx in mismatches})

    first_mismatches: list[dict[str, object]] = []
    for idx in mismatches[:16]:
        linear = int(idx.item())
        lane = linear // _WORDS_PER_LANE
        word = linear % _WORDS_PER_LANE
        first_mismatches.append(
            {
                "linear_word": linear,
                "lane": lane,
                "word": word,
                "lhs": f"0x{int(lhs[linear].item()) & 0xFFFFFFFF:08x}",
                "rhs": f"0x{int(rhs[linear].item()) & 0xFFFFFFFF:08x}",
            }
        )

    lhs_lane = _lane_words(lhs, args.show_lane)
    rhs_lane = _lane_words(rhs, args.show_lane)
    report = {
        "lhs_backend": args.lhs_backend,
        "lhs_consumer": lhs_consumer,
        "lhs_plain_bf16_layout": bool(args.lhs_plain_bf16_layout),
        "rhs_backend": args.rhs_backend,
        "rhs_consumer": rhs_consumer,
        "rhs_plain_bf16_layout": bool(args.rhs_plain_bf16_layout),
        "lanes": _LANES,
        "words_per_lane": _WORDS_PER_LANE,
        "mismatch_count": int(mismatches.numel()),
        "mismatch_lanes": mismatch_lanes,
        "first_mismatches": first_mismatches,
        "show_lane": args.show_lane,
        "lhs_lane_words": _hex_words(lhs_lane, args.show_words),
        "rhs_lane_words": _hex_words(rhs_lane, args.show_words),
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.json_out is not None:
        args.json_out.write_text(text + "\n")


if __name__ == "__main__":
    main()
