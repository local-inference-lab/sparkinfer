"""Regression tests for tp_moe launch arguments."""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import textwrap

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from benchmarks.benchmark_moe import MODEL_PATH, TP_RANK, TP_SIZE, ModelSpec


def _skip_if_no_sm120() -> None:
    if not torch.cuda.is_available():
        pytest.skip("No CUDA")


def _skip_if_unavailable() -> None:
    _skip_if_no_sm120()
    if not MODEL_PATH.exists():
        pytest.skip(f"Model not found at {MODEL_PATH}")
    if not (MODEL_PATH / "model.safetensors.index.json").exists():
        pytest.skip(f"Indexed model weights not found at {MODEL_PATH}")


def _make_spec() -> ModelSpec:
    return ModelSpec(
        hidden_size=4096,
        intermediate_size=1024,
        num_experts=512,
        top_k=10,
        tp_size=TP_SIZE,
        tp_rank=TP_RANK,
    )


def _run_parameter_launch_case(case: str) -> subprocess.CompletedProcess[str]:
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    spec = _make_spec()
    script = textwrap.dedent(
        f"""
        import pathlib
        import sys

        import torch
        from torch.nn import Parameter

        sys.path.insert(0, {str(repo_root)!r})

        from benchmarks.benchmark_moe import ModelSpec, load_expert_weights, make_routed_inputs
        from b12x.integration.tp_moe import (
            allocate_tp_moe_workspace,
            b12x_moe_fp4,
            clear_tp_moe_caches,
        )

        case = {case!r}
        clear_tp_moe_caches()

        device = torch.device("cuda")
        spec = ModelSpec(
            hidden_size={spec.hidden_size},
            intermediate_size={spec.intermediate_size},
            num_experts={spec.num_experts},
            top_k={spec.top_k},
            tp_size={spec.tp_size},
            tp_rank={spec.tp_rank},
        )
        weights = load_expert_weights(pathlib.Path({str(MODEL_PATH)!r}), spec)
        x, topk_ids, topk_weights = make_routed_inputs(spec, 8, seed=123, device=device)

        a1_gscale = weights.w13_input_scale_quant_per_expert.clone()
        a2_gscale = weights.w2_input_scale_quant_per_expert.clone()
        w1_alphas = weights.g1_alphas_per_expert.clone()
        w2_alphas = weights.g2_alphas_per_expert.clone()

        if case in ("scales", "all"):
            a1_gscale = Parameter(a1_gscale, requires_grad=False)
            a2_gscale = Parameter(a2_gscale, requires_grad=False)
        if case in ("alphas", "all"):
            w1_alphas = Parameter(w1_alphas, requires_grad=False)
            w2_alphas = Parameter(w2_alphas, requires_grad=False)

        out = torch.empty_like(x)
        workspace = allocate_tp_moe_workspace(
            x,
            a1_gscale,
            weights.w13_weight,
            a2_gscale,
            weights.w2_weight,
            topk_ids,
            input_scales_static=True,
        )
        print(f"case={{case}} start", flush=True)
        b12x_moe_fp4(
            x,
            a1_gscale,
            weights.w13_weight,
            weights.w13_blockscale_swizzled,
            w1_alphas,
            a2_gscale,
            weights.w2_weight,
            weights.w2_blockscale_swizzled,
            w2_alphas,
            topk_weights,
            topk_ids,
            workspace=workspace,
            output=out,
            input_scales_static=True,
        )
        torch.cuda.synchronize()
        print(f"case={{case}} ok", flush=True)
        """
    )
    env = os.environ.copy()
    env.setdefault("CUTE_DSL_ARCH", "sm_120a")
    env["PYTHONPATH"] = str(repo_root)
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
    )


def _direct_micro_launchable(
    quant_mode: str,
    m: int,
    n: int,
    *,
    weight_E: int = 256,
    k: int = 4096,
    num_topk: int = 10,
) -> bool:
    from b12x.integration.tp_moe import (
        _DIRECT_MICRO_BLOCK_DIM,
        _compiled_direct_micro_accepts_block_dim,
        _get_micro_kernel,
        clear_tp_moe_caches,
    )

    clear_tp_moe_caches()
    torch.empty(1, device="cuda")
    compiled, _ = _get_micro_kernel(
        weight_E,
        m,
        k,
        n,
        num_topk,
        topk_ids_dtype=torch.int32,
        fast_math=True,
        activation="silu",
        quant_mode=quant_mode,
        device=torch.device("cuda"),
    )
    return _compiled_direct_micro_accepts_block_dim(compiled, _DIRECT_MICRO_BLOCK_DIM)


def test_nvfp4_direct_micro_launches_qwen_bs8_shape() -> None:
    _skip_if_no_sm120()

    assert _direct_micro_launchable("nvfp4", 8, 256, weight_E=512)


@pytest.mark.parametrize("case", ["alphas", "scales"])
def test_b12x_moe_accepts_parameter_backed_launch_args(case: str) -> None:
    """The static path should not segfault on Parameter-backed scale tensors."""
    _skip_if_unavailable()

    result = _run_parameter_launch_case(case)
    output = result.stdout + result.stderr
    assert result.returncode == 0, (
        f"child process failed for case={case} with rc={result.returncode}\n{output}"
    )
