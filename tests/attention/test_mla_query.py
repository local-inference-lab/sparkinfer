from __future__ import annotations

import pytest
import torch

from sparkinfer.attention import mla_query
from sparkinfer import gemm
from tests._reference.helpers import require_sparkinfer
from tests.gemm.test_bmm import _make_pack, _rhs_views, _spec


def _inputs(
    *, num_heads: int, m: int, seed: int = 31
) -> tuple[
    torch.Tensor,
    tuple[torch.Tensor, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
]:
    torch.manual_seed(seed)
    values, scales = _make_pack(seed=seed, batch=num_heads)
    weight = _rhs_views(values, scales, batch=num_heads)["n"]
    q_nope = torch.randn(num_heads, m, 192, device="cuda", dtype=torch.bfloat16)
    # Match the real split view: the 64-wide suffix retains a 576-element
    # token/head stride from the full post-RoPE query allocation.
    q_full = torch.randn(m, num_heads, 576, device="cuda", dtype=torch.bfloat16)
    q_pe = q_full[..., 512:]
    q_scale = torch.tensor([0.037], device="cuda", dtype=torch.float32)
    return q_nope, weight, q_pe, q_scale


def _reference_bf16(
    q_nope: torch.Tensor,
    weight: tuple[torch.Tensor, torch.Tensor],
    q_pe: torch.Tensor,
) -> torch.Tensor:
    projected = torch.empty(
        q_nope.shape[0],
        q_nope.shape[1],
        512,
        device=q_nope.device,
        dtype=torch.bfloat16,
    )
    gemm.bmm(q_nope, weight, projected, **_spec("n"))
    return torch.cat((projected.transpose(0, 1), q_pe), dim=-1)


def _reference_fp8(query: torch.Tensor, q_scale: torch.Tensor) -> torch.Tensor:
    inv_scale = torch.ones((), device=query.device, dtype=torch.float32) / q_scale
    return (query.float() * inv_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)


@pytest.mark.parametrize("num_heads", [8, 16])
@pytest.mark.parametrize("m", [1, 6, 9, 32])
@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_fused_query_matches_two_stage_reference(
    num_heads: int, m: int, output_dtype: torch.dtype
) -> None:
    require_sparkinfer()
    q_nope, weight, q_pe, q_scale = _inputs(num_heads=num_heads, m=m)
    reference_bf16 = _reference_bf16(q_nope, weight, q_pe)
    expected = (
        reference_bf16
        if output_dtype == torch.bfloat16
        else _reference_fp8(reference_bf16, q_scale)
    )

    backing = torch.empty(
        m,
        num_heads,
        584,
        device="cuda",
        dtype=output_dtype,
    )
    out = backing[..., :576]
    returned = mla_query.run(q_nope, weight, q_pe, q_scale, out)

    assert returned is out
    if output_dtype == torch.bfloat16:
        assert torch.equal(out, expected)
    else:
        assert torch.equal(out.view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_fused_query_cuda_graph_replays_fresh_inputs(output_dtype: torch.dtype) -> None:
    require_sparkinfer()
    q_nope, weight, q_pe, q_scale = _inputs(num_heads=8, m=4)
    assert mla_query.prewarm(weight, [4], output_dtype=output_dtype) == 1
    out = torch.empty(4, 8, 576, device="cuda", dtype=output_dtype)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        mla_query.run(q_nope, weight, q_pe, q_scale, out)

    fresh_nope = torch.randn_like(q_nope)
    fresh_pe = torch.randn_like(q_pe)
    q_nope.copy_(fresh_nope)
    q_pe.copy_(fresh_pe)
    graph.replay()
    torch.cuda.synchronize()

    reference_bf16 = _reference_bf16(fresh_nope, weight, fresh_pe)
    expected = (
        reference_bf16
        if output_dtype == torch.bfloat16
        else _reference_fp8(reference_bf16, q_scale)
    )
    if output_dtype == torch.bfloat16:
        assert torch.equal(out, expected)
    else:
        assert torch.equal(out.view(torch.uint8), expected.view(torch.uint8))


def test_fused_query_support_gate_is_narrow() -> None:
    device = require_sparkinfer()
    kwargs = dict(
        num_heads=8,
        max_m=32,
        nope_dim=192,
        latent_dim=512,
        output_dtype=torch.bfloat16,
        device=device,
    )
    assert mla_query.can_implement(**kwargs)
    assert mla_query.can_implement(**{**kwargs, "output_dtype": torch.float8_e4m3fn})
    assert not mla_query.can_implement(**{**kwargs, "num_heads": 11})
    assert not mla_query.can_implement(**{**kwargs, "max_m": 33})
    assert not mla_query.can_implement(**{**kwargs, "nope_dim": 256})


def test_fused_query_rejects_wrong_rope_layout() -> None:
    require_sparkinfer()
    q_nope, weight, q_pe, q_scale = _inputs(num_heads=8, m=2)
    out = torch.empty(2, 8, 576, device="cuda", dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="q_pe must have shape"):
        mla_query.run(q_nope, weight, q_pe[..., :32], q_scale, out)
