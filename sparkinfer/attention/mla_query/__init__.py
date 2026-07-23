"""Fused SM12x MLA absorbed-query assembly.

The operation multiplies BF16 ``q_nope`` by the native rowwise-MXFP8
``W_UK_T`` pack and writes the result directly into a token-major MLA query.
It appends the existing 64-wide RoPE query in the same launch.  The caller may
request BF16 output, or static per-tensor E4M3 output for an FP8 attention
backend.  FP8 mode deliberately rounds the GEMM result through BF16 before
scaling, preserving the established two-kernel numerical contract.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..._lib.meta import OpMeta, Provenance, install_lazy_api

META = OpMeta(
    name="mla_query",
    group="attention",
    api_style="oneshot",
    entry_points=(
        "run",
        "prewarm",
        "can_implement",
        "is_supported",
        "clear_caches",
    ),
    dtypes=("bf16", "fp8_e4m3"),
    recipes=("glm_nsa",),
    provenance=Provenance(
        repo="https://github.com/local-inference-lab/sparkinfer",
        commit="1a88b389",
        paths=("sparkinfer/gemm/_bmm/_mxfp8.py",),
    ),
    test_path="tests/attention/test_mla_query.py",
    since="1.0.1",
    notes="Fused MXFP8 query BMM, RoPE append, and optional static E4M3 quant.",
)

if TYPE_CHECKING:
    from .api import (  # noqa: F401
        can_implement,
        clear_caches,
        is_supported,
        prewarm,
        run,
    )

install_lazy_api(globals(), META)
