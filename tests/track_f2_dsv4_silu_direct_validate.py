#!/usr/bin/env python3
"""Focused docker validation for Track F2 DSv4 SiLU direct MoE dispatch.

This is intentionally a runtime helper rather than a pytest-only test: it is
meant to run inside the b12x/cutlass docker image before any model C1/C8 sweep.
It compares the generic b12x MoE path with B12X_DSV4_SILU_DIRECT=0 against the
exact DSv4 direct path with B12X_DSV4_SILU_DIRECT=1 on live decode dimensions.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import logging
import os
import pathlib
import sys
import time

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from b12x.cute.fp4 import swizzle_block_scale
from b12x.integration.tp_moe import (
    allocate_tp_moe_workspace_pool,
    b12x_moe_fp4,
    clear_tp_moe_caches,
    get_tp_moe_debug_counters,
)


EXPECTED_LOG_FRAGMENT = "bypassing generic flashinfer/backend dispatch"


@contextlib.contextmanager
def _env(name: str, value: str):
    old = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old


def _make_scale_storage(experts: int, rows: int, cols: int, *, device: torch.device) -> torch.Tensor:
    cols_blocks = cols // 16
    scale = torch.full(
        (experts, rows, cols_blocks),
        0.015625,
        dtype=torch.float32,
        device=device,
    ).to(torch.float8_e4m3fn)
    return swizzle_block_scale(scale)


def make_case(tokens: int, *, experts: int, hidden: int, intermediate: int, topk: int, seed: int):
    torch.manual_seed(seed + tokens)
    device = torch.device("cuda")
    x = (torch.randn(tokens, hidden, dtype=torch.float32, device=device) * 0.02).to(torch.bfloat16)
    logits = torch.randn(tokens, experts, dtype=torch.float32, device=device)
    topk_logits, topk_ids = torch.topk(logits, topk, dim=-1)
    topk_weights = torch.softmax(topk_logits, dim=-1).contiguous()
    topk_ids = topk_ids.to(torch.int32).contiguous()

    # Exact DSv4 TP=4 tensor shapes:
    # w1: [64, 4096, 2048] uint8, w2: [64, 4096, 1024] uint8.
    w1_fp4 = torch.randint(
        0,
        256,
        (experts, 2 * intermediate, hidden // 2),
        dtype=torch.uint8,
        device=device,
    )
    w2_fp4 = torch.randint(
        0,
        256,
        (experts, hidden, intermediate // 2),
        dtype=torch.uint8,
        device=device,
    )
    w1_blockscale = _make_scale_storage(experts, 2 * intermediate, hidden, device=device)
    w2_blockscale = _make_scale_storage(experts, hidden, intermediate, device=device)
    w1_alphas = torch.full((experts,), 0.25, dtype=torch.float32, device=device)
    w2_alphas = torch.full((experts,), 0.25, dtype=torch.float32, device=device)
    a1_gscale = torch.ones((), dtype=torch.float32, device=device)
    a2_gscale = torch.ones((), dtype=torch.float32, device=device)
    return (
        x,
        a1_gscale,
        w1_fp4,
        w1_blockscale,
        w1_alphas,
        a2_gscale,
        w2_fp4,
        w2_blockscale,
        w2_alphas,
        topk_weights,
        topk_ids,
    )


def run_once(case, *, direct: bool, output: torch.Tensor | None = None):
    workspace = allocate_tp_moe_workspace_pool()
    with _env("B12X_DSV4_SILU_DIRECT", "1" if direct else "0"):
        return b12x_moe_fp4(
            *case,
            workspace=workspace,
            output=output,
            input_scales_are_reciprocal=True,
            input_scales_static=True,
            activation="silu",
        )


def wall_ms(fn, *, warmup: int, reps: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(reps):
        fn()
        torch.cuda.synchronize()
    end = time.perf_counter()
    return (end - start) * 1000.0 / reps


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", default="1,2,4,8")
    parser.add_argument("--experts", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=2048)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260428)
    parser.add_argument("--rtol", type=float, default=0.0)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--min-speedup", type=float, default=1.0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("FAIL: CUDA is unavailable", file=sys.stderr)
        return 2

    logging.basicConfig(level=logging.WARNING)
    tokens_list = [int(x) for x in args.tokens.split(",") if x.strip()]
    print(
        "Track F2 DSv4 SiLU direct validation "
        f"E={args.experts} H={args.hidden} I={args.intermediate} topk={args.topk}"
    )
    print(f"expected_log_fragment={EXPECTED_LOG_FRAGMENT!r}")

    failures: list[str] = []
    for tokens in tokens_list:
        clear_tp_moe_caches()
        case = make_case(
            tokens,
            experts=args.experts,
            hidden=args.hidden,
            intermediate=args.intermediate,
            topk=args.topk,
            seed=args.seed,
        )

        generic = run_once(case, direct=False).detach().clone()
        torch.cuda.synchronize()

        clear_tp_moe_caches()
        log_stream = io.StringIO()
        handler = logging.StreamHandler(log_stream)
        logger = logging.getLogger("b12x.integration.tp_moe")
        logger.addHandler(handler)
        try:
            direct = run_once(case, direct=True).detach().clone()
            torch.cuda.synchronize()
        finally:
            logger.removeHandler(handler)

        max_abs = (generic.float() - direct.float()).abs().max().item()
        same = torch.allclose(generic, direct, rtol=args.rtol, atol=args.atol)
        counters = get_tp_moe_debug_counters()
        log_text = log_stream.getvalue()
        fired = counters.get("dsv4_silu_direct_hits", 0) > 0
        logged = EXPECTED_LOG_FRAGMENT in log_text

        generic_out = torch.empty_like(case[0])
        direct_out = torch.empty_like(case[0])
        clear_tp_moe_caches()
        generic_ms = wall_ms(
            lambda: run_once(case, direct=False, output=generic_out),
            warmup=args.warmup,
            reps=args.reps,
        )
        clear_tp_moe_caches()
        direct_ms = wall_ms(
            lambda: run_once(case, direct=True, output=direct_out),
            warmup=args.warmup,
            reps=args.reps,
        )
        speedup = generic_ms / direct_ms if direct_ms > 0 else float("inf")

        print(
            f"tokens={tokens} generic_ms={generic_ms:.4f} direct_ms={direct_ms:.4f} "
            f"speedup={speedup:.3f} max_abs={max_abs:.6g} "
            f"direct_hits={counters.get('dsv4_silu_direct_hits', 0)} logged={logged}"
        )

        if not same:
            failures.append(f"tokens={tokens}: direct output differs from generic, max_abs={max_abs:.6g}")
        if not fired:
            failures.append(f"tokens={tokens}: dsv4_silu_direct_hits did not increment")
        if not logged:
            failures.append(f"tokens={tokens}: missing log fragment {EXPECTED_LOG_FRAGMENT!r}")
        if speedup < args.min_speedup:
            failures.append(
                f"tokens={tokens}: direct speedup {speedup:.3f} < required {args.min_speedup:.3f}"
            )

    if failures:
        print("FAIL:")
        for failure in failures:
            print(f"  {failure}")
        return 1

    print("PASS: direct path fired, matched generic output, and met microbench threshold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
