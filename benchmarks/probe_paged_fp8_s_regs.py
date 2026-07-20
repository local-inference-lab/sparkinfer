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

from sparkinfer.integration.attention import PagedAttentionWorkspace, clear_attention_caches
from tests.test_attention_paged_planner import _make_inputs
from tests.test_paged_attention_workspace_api import _quantize_paged_kv_cache_e4m3

_LANES = 32
_WORDS_PER_LANE = 8
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


def _dump_words(*, use_tma: bool) -> torch.Tensor:
    overrides = {
        "SPARKINFER_PAGED_KV_DEBUG_DUMP": "SREGS",
        "SPARKINFER_PAGED_KV_TMA": "1" if use_tma else None,
        "SPARKINFER_PAGED_KV_TMA_FP8_RAW_ISSUE": "1" if use_tma else None,
    }
    with _env(overrides):
        clear_attention_caches()
        torch.manual_seed(_SEED)
        q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
            q_seqlens=[1] * 8,
            cache_seqlens=[64] * 8,
            q_heads=16,
            kv_heads=8,
            head_dim_qk=256,
            head_dim_vo=256,
            dtype=torch.bfloat16,
            kv_dtype=torch.bfloat16,
        )
        k_fp8, v_fp8, k_descale, v_descale = _quantize_paged_kv_cache_e4m3(
            k_cache,
            v_cache,
            page_table,
            cache_seqlens,
        )
        workspace = PagedAttentionWorkspace.for_tensors(
            mode="decode",
            q=q,
            k_cache=k_fp8,
            v_cache=v_fp8,
        )
        workspace.prepare(page_table, cache_seqlens, cu_seqlens_q, disable_split_kv=True)
        output = torch.zeros_like(q)
        workspace.run(
            q,
            k_fp8,
            v_fp8,
            output=output,
            k_descale=k_descale,
            v_descale=v_descale,
        )
        torch.cuda.synchronize()
    raw8 = output.view(torch.uint8).reshape(-1).to(torch.int64)
    words = raw8[0::4] | (raw8[1::4] << 8) | (raw8[2::4] << 16) | (raw8[3::4] << 24)
    return words[: _LANES * _WORDS_PER_LANE].cpu()


def _lane_words(words: torch.Tensor, lane: int) -> list[int]:
    start = lane * _WORDS_PER_LANE
    end = start + _WORDS_PER_LANE
    return [int(v) for v in words[start:end].tolist()]


def _hex_words(words: list[int], count: int) -> list[str]:
    return [f"0x{value & 0xFFFFFFFF:08x}" for value in words[:count]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare raw FP8 QK score fragment registers for paged attention.")
    parser.add_argument("--show-lane", type=int, default=0)
    parser.add_argument("--show-words", type=int, default=8)
    args = parser.parse_args()

    lhs = _dump_words(use_tma=False)
    rhs = _dump_words(use_tma=True)
    mismatches = (lhs != rhs).nonzero(as_tuple=False).reshape(-1)
    mismatch_lanes = sorted({int(idx.item()) // _WORDS_PER_LANE for idx in mismatches})
    first_mismatches = []
    for idx in mismatches[:16]:
        linear = int(idx.item())
        first_mismatches.append(
            {
                "linear_word": linear,
                "lane": linear // _WORDS_PER_LANE,
                "word": linear % _WORDS_PER_LANE,
                "lhs": f"0x{int(lhs[linear].item()) & 0xFFFFFFFF:08x}",
                "rhs": f"0x{int(rhs[linear].item()) & 0xFFFFFFFF:08x}",
            }
        )

    report = {
        "lanes": _LANES,
        "words_per_lane": _WORDS_PER_LANE,
        "mismatch_count": int(mismatches.numel()),
        "mismatch_lanes": mismatch_lanes,
        "first_mismatches": first_mismatches,
        "show_lane": args.show_lane,
        "cpasync_lane_words": _hex_words(_lane_words(lhs, args.show_lane), args.show_words),
        "tma_lane_words": _hex_words(_lane_words(rhs, args.show_lane), args.show_words),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
