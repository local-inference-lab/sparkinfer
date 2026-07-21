"""Private implementation package for :func:`sparkinfer.gemm.bmm`.

``bmm`` dispatches from dtype and layout metadata.  The MXFP8 specialization
multiplies BF16 activations by a rowwise-MXFP8 right-hand side and writes BF16
output.  It supports K-major and N-major zero-copy views.

The launch path is workspace- and allocation-free.  Call ``prewarm_bmm`` with
every M that can appear inside a CUDA graph; a compile miss during capture
raises rather than recording an invalid graph.

Example::

    from sparkinfer import gemm

    spec = dict(
        a_dtype="bfloat16",
        b_dtype="float8_e4m3fn",
        sf_dtype="float8_e8m0fnu",
        c_dtype="bfloat16",
        sf_vec_size=32,
        b_major="k",
        sf_axis="k",
    )
    gemm.prewarm_bmm((b_values, b_scales), (1, 4, 8), **spec)
    gemm.bmm(lhs, (b_values, b_scales), out, **spec)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..._lib.meta import OpMeta, Provenance, install_lazy_api

META = OpMeta(
    name="bmm",
    group="gemm",
    api_style="oneshot",
    entry_points=(
        "bmm",
        "prewarm_bmm",
        "can_implement_bmm",
        "is_bmm_supported",
    ),
    dtypes=("bf16", "fp8_e4m3"),
    recipes=("mxfp8",),
    provenance=Provenance(
        repo="https://github.com/MadeBy561/b12x",
        commit="3854a4b1",
        paths=("b12x/gemm/qbmm_absorb.py",),
    ),
    test_path="tests/gemm/test_bmm.py",
    since="1.0.1",
    notes="Generic BMM API with a BF16 x rowwise-MXFP8 backend.",
)

if TYPE_CHECKING:  # static analysis only; runtime resolution is lazy
    from .api import (  # noqa: F401
        bmm,
        can_implement_bmm,
        is_bmm_supported,
        prewarm_bmm,
    )

install_lazy_api(globals(), META)
