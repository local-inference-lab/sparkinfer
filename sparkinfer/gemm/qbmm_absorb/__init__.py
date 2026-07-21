"""Weight-only MXFP8 batched GEMM for MLA absorbed projections (one-shot).

Decode-path MLA consumes the absorbed projections directly from the kv_b
mxfp8 pack (fp8 e4m3 values + per-32-in-axis e8m0 scales), replacing the
resident BF16 ``W_UK_T``/``W_UV`` pair.  One kernel template, two
orientations (``ukt``/``uv``); dequant is bitwise the e8m0-viewed bf16-chain
torch reference; bf16 HMMA m16n8k16 with fp32 accumulators, K unsplit, no
workspace, no allocation on the launch path.  ``warmup_qbmm_absorb`` must
precompile every CUDA-graph-visible batch size before capture -- a compile
miss during stream capture raises instead of corrupting the graph.

Example:
    from sparkinfer.gemm import qbmm_absorb as qbmm

    qbmm.warmup_qbmm_absorb(pack_values, pack_scales)        # before capture
    # out[n, b, :] = a[n, b, :] @ W_UK_T[n], dequantized on the fly
    qbmm.qbmm_absorb_ukt(a, pack_values, pack_scales, out)
    # out[n, b, :] = a[n, b, :] @ W_UV[n]; call-site view of (B, N, V) storage
    qbmm.qbmm_absorb_uv(a2, pack_values, pack_scales, out2.transpose(0, 1))
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..._lib.meta import OpMeta, Provenance, install_lazy_api

META = OpMeta(
    name="qbmm_absorb",
    group="gemm",
    api_style="oneshot",
    entry_points=(
        "QbmmAbsorbKernel",
        "qbmm_absorb",
        "qbmm_absorb_ukt",
        "qbmm_absorb_uv",
        "qbmm_absorb_supported",
        "warmup_qbmm_absorb",
        "is_supported",
    ),
    dtypes=("bf16",),
    recipes=("mxfp8",),
    provenance=Provenance(
        repo="https://github.com/MadeBy561/b12x",
        commit="3854a4b1",
        paths=("b12x/gemm/qbmm_absorb.py",),
    ),
    test_path="tests/gemm/test_qbmm_absorb.py",
    since="1.0.1",
)

if TYPE_CHECKING:  # static analysis only; runtime resolution is lazy
    from .api import (  # noqa: F401
        QbmmAbsorbKernel,
        is_supported,
        qbmm_absorb,
        qbmm_absorb_supported,
        qbmm_absorb_ukt,
        qbmm_absorb_uv,
        warmup_qbmm_absorb,
    )

install_lazy_api(globals(), META)
