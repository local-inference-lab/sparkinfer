"""Planned full-rotation EXL3 Trellis MoE for SM12x.

This op owns the production ``trellis3_t256`` MCG lifecycle. Weight preparation
wraps projection-major native EXL3 tensors without repacking. Planning fixes the
token, route, tile, and exact M-block capacities and eagerly compiles both the
fused MoE launch and full-rotation FP32 top-k reductions. Binding maps one
caller-owned uint8 scratch arena into stable views; ``run`` performs no tensor
allocation and is CUDA-graph-capture safe after ordinary eager warmup.

Example:
    from sparkinfer.moe import trellis_moe

    weights = trellis_moe.prepare_weights(
        w13, w2,
        gate_suh=gate_suh,
        up_suh=up_suh,
        intermediate_rotations=intermediate_rotations,
        down_svh=down_svh,
        codebook="mcg",
    )
    plan = trellis_moe.plan(trellis_moe.Caps(
        max_tokens=32,
        num_topk=8,
        num_experts=weights.num_experts,
        hidden_size=weights.hidden_size,
        intermediate_size=weights.intermediate_size,
        route_num_experts=256,
        block_size_m=8,
        input_dtype=torch.bfloat16,
        device=weights.device,
    ))
    spec = plan.scratch_specs()[0]
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    binding = trellis_moe.bind(
        plan,
        scratch=scratch,
        a=x,
        weights=weights,
        topk_weights=router_weights,
        topk_ids=router_ids,
    )
    output = trellis_moe.run(binding=binding)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..._lib.meta import OpMeta, Provenance, install_lazy_api

META = OpMeta(
    name="trellis_moe",
    group="moe",
    api_style="planned",
    entry_points=(
        "Caps",
        "Plan",
        "Weights",
        "Binding",
        "plan",
        "prepare_weights",
        "bind",
        "run",
        "is_supported",
        "clear_caches",
    ),
    dtypes=("bf16", "fp16"),
    recipes=("trellis3_t256_mcg",),
    requires=("triton",),
    provenance=Provenance(
        repo="https://github.com/brandonmmusic-max/b12x",
        commit="e611971",
        paths=("b12x/moe/fused/w4a16/",),
    ),
    test_path="tests/moe/test_trellis_moe.py",
    since="1.1.0",
    notes="Projection-major MCG Trellis weights with fused full rotations.",
)

if TYPE_CHECKING:  # static analysis only; runtime resolution is lazy
    from .api import (  # noqa: F401
        Binding,
        Caps,
        Plan,
        Weights,
        bind,
        clear_caches,
        is_supported,
        plan,
        prepare_weights,
        run,
    )

install_lazy_api(globals(), META)
