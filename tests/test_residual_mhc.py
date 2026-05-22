from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from b12x.integration import (
    b12x_mhc_post,
    b12x_mhc_pre,
    empty_mhc_workspace,
)

from .helpers import require_sm120


def _mhc_pre_reference(
    residual: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
    *,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flat = residual.flatten(1).float()
    mixes = F.linear(flat, fn) * torch.rsqrt(
        flat.square().mean(dim=-1, keepdim=True) + rms_eps
    )
    pre = torch.sigmoid(mixes[:, :4] * scale[0] + bias[:4]) + hc_eps
    post = 2 * torch.sigmoid(mixes[:, 4:8] * scale[1] + bias[4:8])
    comb = mixes[:, 8:].view(-1, 4, 4) * scale[2] + bias[8:].view(4, 4)
    comb = torch.softmax(comb, dim=-1) + hc_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + hc_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)
    y = (pre.unsqueeze(-1) * residual.float()).sum(dim=1).to(residual.dtype)
    return y, post, comb


def _mhc_post_reference(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    return (
        post.unsqueeze(-1) * x.unsqueeze(1).float()
        + (comb.unsqueeze(-1) * residual.unsqueeze(2).float()).sum(dim=1)
    ).to(x.dtype)


def _make_inputs(
    *,
    tokens: int,
    hidden_size: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    residual = (
        torch.randn((tokens, 4, hidden_size), generator=gen, dtype=torch.float32).to(device)
        / 3
    ).to(torch.bfloat16)
    x = (
        torch.randn((tokens, hidden_size), generator=gen, dtype=torch.float32).to(device)
        / 4
    ).to(torch.bfloat16)
    fn = torch.randn((24, 4 * hidden_size), generator=gen, dtype=torch.float32).to(device) / 64
    scale = torch.randn((3,), generator=gen, dtype=torch.float32).to(device) / 3
    bias = torch.randn((24,), generator=gen, dtype=torch.float32).to(device) / 5
    return residual.contiguous(), x.contiguous(), fn.contiguous(), scale.contiguous(), bias.contiguous()


@pytest.mark.parametrize("tokens", [1, 3])
def test_b12x_mhc_pre_post_match_reference(tokens: int) -> None:
    device = require_sm120()
    hidden_size = 4096
    residual, x, fn, scale, bias = _make_inputs(
        tokens=tokens,
        hidden_size=hidden_size,
        seed=91_400 + tokens,
        device=device,
    )
    workspace = empty_mhc_workspace(
        num_tokens=tokens,
        hidden_size=hidden_size,
        device=device,
    )
    y, post, comb = b12x_mhc_pre(
        residual,
        fn,
        scale,
        bias,
        rms_eps=1e-6,
        hc_eps=1e-6,
        sinkhorn_iters=20,
        workspace=workspace,
    )
    out = b12x_mhc_post(x, residual, post, comb, workspace=workspace)
    torch.cuda.synchronize(device)

    y_ref, post_ref, comb_ref = _mhc_pre_reference(
        residual,
        fn,
        scale,
        bias,
        rms_eps=1e-6,
        hc_eps=1e-6,
        sinkhorn_iters=20,
    )
    out_ref = _mhc_post_reference(x, residual, post, comb)
    torch.testing.assert_close(y, y_ref, rtol=0.0, atol=2e-3)
    torch.testing.assert_close(post, post_ref, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(comb, comb_ref, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(out, out_ref, rtol=0.0, atol=2e-2)


def test_b12x_mhc_pre_post_graph_capture() -> None:
    device = require_sm120()
    tokens = 2
    hidden_size = 4096
    residual, x, fn, scale, bias = _make_inputs(
        tokens=tokens,
        hidden_size=hidden_size,
        seed=91_410,
        device=device,
    )
    workspace = empty_mhc_workspace(
        num_tokens=tokens,
        hidden_size=hidden_size,
        device=device,
    )
    y = workspace.y
    post = workspace.post
    comb = workspace.comb
    out = workspace.out

    def run() -> None:
        b12x_mhc_pre(
            residual,
            fn,
            scale,
            bias,
            rms_eps=1e-6,
            hc_eps=1e-6,
            sinkhorn_iters=20,
            workspace=workspace,
            y_out=y,
            post_out=post,
            comb_out=comb,
        )
        b12x_mhc_post(x, residual, post, comb, workspace=workspace, out=out)

    run()
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    torch.cuda.synchronize(device)

    y_ref, post_ref, comb_ref = _mhc_pre_reference(
        residual,
        fn,
        scale,
        bias,
        rms_eps=1e-6,
        hc_eps=1e-6,
        sinkhorn_iters=20,
    )
    out_ref = _mhc_post_reference(x, residual, post, comb)
    torch.testing.assert_close(y, y_ref, rtol=0.0, atol=2e-3)
    torch.testing.assert_close(post, post_ref, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(comb, comb_ref, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(out, out_ref, rtol=0.0, atol=2e-2)


def test_b12x_mhc_pre_rejects_single_cta_contract() -> None:
    device = require_sm120()
    residual, _, fn, scale, bias = _make_inputs(
        tokens=1,
        hidden_size=4096,
        seed=91_420,
        device=device,
    )
    with pytest.raises(ValueError, match="no single-CTA fallback"):
        b12x_mhc_pre(
            residual,
            fn,
            scale,
            bias,
            rms_eps=1e-6,
            hc_eps=1e-6,
            sinkhorn_iters=20,
            split_k=1,
        )
