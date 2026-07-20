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

_PAGE_SIZE = 64
_HEAD_DIM = 256
_NUM_WORDS = _PAGE_SIZE * _HEAD_DIM // 2
_DUMP_Q = 8
_DUMP_WORD_CAPACITY = _DUMP_Q * 8 * (_HEAD_DIM // 2)
_WORDS_PER_ROW = _HEAD_DIM // 2
_CHUNKS_PER_ROW = _HEAD_DIM // 8
_WORDS_PER_CHUNK = 4


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


def _build_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
        q_seqlens=[1],
        cache_seqlens=[64],
        dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
    )
    q.zero_()
    k_cache.zero_()
    v_u16 = v_cache.view(torch.uint16)
    v_u16.copy_(
        torch.arange(v_u16.numel(), device=v_u16.device, dtype=torch.int32).to(torch.uint16).reshape(v_u16.shape)
    )
    return q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q


def _dump_words(*, use_tma: bool) -> torch.Tensor:
    overrides = {
        "SPARKINFER_PAGED_KV_DEBUG_DUMP": "SVWORDS",
        "SPARKINFER_PAGED_KV_TMA": "1" if use_tma else None,
        "SPARKINFER_PAGED_KV_TMA_COPYFRAG_QK": None,
        "SPARKINFER_PAGED_KV_TMA_COPYFRAG_PV": None,
        "SPARKINFER_PAGED_KV_TMA_DONOR_GEMM": None,
    }
    with _env(overrides):
        clear_attention_caches()
        q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _build_inputs()
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
        output = torch.zeros((_DUMP_Q, 8, _HEAD_DIM), dtype=torch.bfloat16, device=q.device)
        workspace.run(q, k_cache, v_cache, output=output)
        torch.cuda.synchronize()
    raw8 = output.view(torch.uint8).reshape(-1).to(torch.int64)
    words = raw8[0::4] | (raw8[1::4] << 8) | (raw8[2::4] << 16) | (raw8[3::4] << 24)
    return words[:_NUM_WORDS].cpu()


def _pairs(words: torch.Tensor, count: int) -> list[dict[str, int]]:
    pairs: list[dict[str, int]] = []
    for flat_idx, packed in enumerate(words[:count].tolist()):
        value = int(packed)
        pairs.append(
            {
                "flat_idx": flat_idx,
                "lo": value & 0xFFFF,
                "hi": (value >> 16) & 0xFFFF,
            }
        )
    return pairs


def _inverse(words: torch.Tensor) -> list[int]:
    inverse = [-1] * _NUM_WORDS
    for flat_idx, packed in enumerate(words.tolist()):
        value = int(packed)
        lo = value & 0xFFFF
        hi = (value >> 16) & 0xFFFF
        if hi == lo + 1 and (lo & 1) == 0:
            inverse[lo // 2] = flat_idx
    return inverse


def _row_chunk_sources(words: torch.Tensor, row: int) -> list[dict[str, int]]:
    if row < 0 or row >= _PAGE_SIZE:
        raise ValueError(f"row must be in [0, {_PAGE_SIZE - 1}]")
    row_words = words[row * _WORDS_PER_ROW : (row + 1) * _WORDS_PER_ROW]
    chunks: list[dict[str, int]] = []
    for chunk_idx in range(_CHUNKS_PER_ROW):
        chunk_words = row_words[chunk_idx * _WORDS_PER_CHUNK : (chunk_idx + 1) * _WORDS_PER_CHUNK]
        first = int(chunk_words[0].item())
        lo = first & 0xFFFF
        src_linear = lo
        chunks.append(
            {
                "chunk": chunk_idx,
                "src_row": src_linear // _HEAD_DIM,
                "src_chunk": (src_linear % _HEAD_DIM) // 8,
            }
        )
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dump the live paged BF16 shared-word layout for the staged V tile on cp.async and TMA paths."
    )
    parser.add_argument(
        "--show-words",
        type=int,
        default=32,
        help="number of flat words to include from each path",
    )
    parser.add_argument(
        "--json-out",
        type=pathlib.Path,
        default=None,
        help="optional path to write the full JSON report",
    )
    parser.add_argument(
        "--show-rows",
        type=str,
        default="0,1,2,7,8",
        help="comma-separated staged rows to include as 16B-chunk source maps",
    )
    args = parser.parse_args()

    if _DUMP_WORD_CAPACITY < _NUM_WORDS:
        raise RuntimeError("debug output buffer is too small for the staged V tile dump")

    cpasync_words = _dump_words(use_tma=False)
    tma_words = _dump_words(use_tma=True)
    show_rows = [int(part) for part in args.show_rows.split(",") if part]
    mismatches = (cpasync_words != tma_words).nonzero(as_tuple=False).reshape(-1)
    report = {
        "num_words": _NUM_WORDS,
        "mismatch_count": int(mismatches.numel()),
        "first_mismatches": [
            {
                "flat_idx": int(idx.item()),
                "cpasync": f"0x{int(cpasync_words[int(idx.item())].item()) & 0xFFFFFFFF:08x}",
                "tma": f"0x{int(tma_words[int(idx.item())].item()) & 0xFFFFFFFF:08x}",
            }
            for idx in mismatches[:16]
        ],
        "cpasync_first_pairs": _pairs(cpasync_words, args.show_words),
        "tma_first_pairs": _pairs(tma_words, args.show_words),
        "cpasync_first_inverse": _inverse(cpasync_words)[:64],
        "tma_first_inverse": _inverse(tma_words)[:64],
        "cpasync_row_chunk_sources": {
            str(row): _row_chunk_sources(cpasync_words, row) for row in show_rows
        },
        "tma_row_chunk_sources": {
            str(row): _row_chunk_sources(tma_words, row) for row in show_rows
        },
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.json_out is not None:
        args.json_out.write_text(text + "\n")


if __name__ == "__main__":
    main()
