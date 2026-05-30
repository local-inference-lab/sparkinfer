from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from b12x.integration import (
    b12x_mhc_post,
    b12x_mhc_post_pre,
    b12x_mhc_pre,
    b12x_mhc_pre_post,
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


def _rms_norm_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    x_float = x.float()
    return (
        x_float
        * torch.rsqrt(x_float.square().mean(dim=-1, keepdim=True) + eps)
        * weight.float()
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
def test_b12x_mhc_separate_pre_post_match_reference(tokens: int) -> None:
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
    torch.testing.assert_close(y, y_ref, rtol=0.0, atol=4e-3)
    torch.testing.assert_close(post, post_ref, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(comb, comb_ref, rtol=2e-6, atol=2e-6)

    out = b12x_mhc_post(x, residual, post, comb, workspace=workspace)
    torch.cuda.synchronize(device)

    out_ref = _mhc_post_reference(x, residual, post, comb)
    torch.testing.assert_close(out, out_ref, rtol=0.0, atol=2e-2)


@pytest.mark.parametrize("tokens", [1, 3])
@pytest.mark.parametrize("norm_dtype", [torch.bfloat16, torch.float32])
def test_b12x_mhc_pre_with_fused_rmsnorm_match_reference(
    tokens: int,
    norm_dtype: torch.dtype,
) -> None:
    device = require_sm120()
    hidden_size = 4096
    residual, _, fn, scale, bias = _make_inputs(
        tokens=tokens,
        hidden_size=hidden_size,
        seed=91_420 + tokens,
        device=device,
    )
    norm_gen = torch.Generator(device="cpu")
    norm_gen.manual_seed(91_421 + tokens)
    norm_weight = (
        torch.randn((hidden_size,), generator=norm_gen, dtype=torch.float32)
        .to(device)
        .to(norm_dtype)
        .contiguous()
    )

    y, post, comb = b12x_mhc_pre(
        residual,
        fn,
        scale,
        bias,
        rms_eps=1e-6,
        hc_eps=1e-6,
        sinkhorn_iters=20,
        norm_weight=norm_weight,
        norm_eps=1e-6,
    )
    torch.cuda.synchronize(device)

    y_raw_ref, post_ref, comb_ref = _mhc_pre_reference(
        residual,
        fn,
        scale,
        bias,
        rms_eps=1e-6,
        hc_eps=1e-6,
        sinkhorn_iters=20,
    )
    y_ref = _rms_norm_reference(y_raw_ref, norm_weight, 1e-6)
    torch.testing.assert_close(y, y_ref, rtol=0.0, atol=5e-3)
    torch.testing.assert_close(post, post_ref, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(comb, comb_ref, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("tokens", [1, 3])
def test_b12x_mhc_fused_pre_post_match_reference(tokens: int) -> None:
    device = require_sm120()
    hidden_size = 4096
    residual, x, fn, scale, bias = _make_inputs(
        tokens=tokens,
        hidden_size=hidden_size,
        seed=91_430 + tokens,
        device=device,
    )
    workspace = empty_mhc_workspace(
        num_tokens=tokens,
        hidden_size=hidden_size,
        device=device,
    )

    y, post, comb, out = b12x_mhc_pre_post(
        x,
        residual,
        fn,
        scale,
        bias,
        rms_eps=1e-6,
        hc_eps=1e-6,
        sinkhorn_iters=20,
        workspace=workspace,
    )
    torch.cuda.synchronize(device)

    assert y.untyped_storage().data_ptr() == workspace.y.untyped_storage().data_ptr()
    assert post.untyped_storage().data_ptr() == workspace.post.untyped_storage().data_ptr()
    assert comb.untyped_storage().data_ptr() == workspace.comb.untyped_storage().data_ptr()
    assert out.untyped_storage().data_ptr() == workspace.out.untyped_storage().data_ptr()

    y_ref, post_ref, comb_ref = _mhc_pre_reference(
        residual,
        fn,
        scale,
        bias,
        rms_eps=1e-6,
        hc_eps=1e-6,
        sinkhorn_iters=20,
    )
    out_ref = _mhc_post_reference(x, residual, post_ref, comb_ref)
    torch.testing.assert_close(y, y_ref, rtol=0.0, atol=2e-3)
    torch.testing.assert_close(post, post_ref, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(comb, comb_ref, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(out, out_ref, rtol=0.0, atol=2e-2)


@pytest.mark.parametrize("tokens", [1, 3, 8])
def test_b12x_mhc_fused_post_pre_match_reference(tokens: int) -> None:
    device = require_sm120()
    hidden_size = 4096
    residual, x, fn, scale, bias = _make_inputs(
        tokens=tokens,
        hidden_size=hidden_size,
        seed=91_450 + tokens,
        device=device,
    )
    workspace = empty_mhc_workspace(
        num_tokens=tokens,
        hidden_size=hidden_size,
        device=device,
    )
    _, prev_post, prev_comb = _mhc_pre_reference(
        residual,
        fn,
        scale,
        bias,
        rms_eps=1e-6,
        hc_eps=1e-6,
        sinkhorn_iters=20,
    )
    prev_post_arg = prev_post.contiguous()
    if tokens == 3:
        prev_post_arg = prev_post_arg.unsqueeze(-1).contiguous()

    residual_cur, post, comb, y = b12x_mhc_post_pre(
        x,
        residual,
        prev_post_arg,
        prev_comb.contiguous(),
        fn,
        scale,
        bias,
        rms_eps=1e-6,
        hc_eps=1e-6,
        sinkhorn_iters=20,
        workspace=workspace,
    )
    torch.cuda.synchronize(device)

    assert residual_cur.untyped_storage().data_ptr() == workspace.out.untyped_storage().data_ptr()
    assert post.untyped_storage().data_ptr() == workspace.post.untyped_storage().data_ptr()
    assert comb.untyped_storage().data_ptr() == workspace.comb.untyped_storage().data_ptr()
    assert y.untyped_storage().data_ptr() == workspace.y.untyped_storage().data_ptr()

    residual_ref = _mhc_post_reference(x, residual, prev_post, prev_comb)
    y_ref, post_ref, comb_ref = _mhc_pre_reference(
        residual_ref,
        fn,
        scale,
        bias,
        rms_eps=1e-6,
        hc_eps=1e-6,
        sinkhorn_iters=20,
    )
    torch.testing.assert_close(residual_cur, residual_ref, rtol=0.0, atol=2e-2)
    torch.testing.assert_close(y, y_ref, rtol=0.0, atol=4e-3)
    scalar_atol = 2e-5 if tokens >= 8 else 1e-5
    torch.testing.assert_close(post, post_ref, rtol=2e-6, atol=scalar_atol)
    torch.testing.assert_close(comb, comb_ref, rtol=2e-6, atol=scalar_atol)


@pytest.mark.parametrize("tokens", [1, 3])
def test_b12x_mhc_fused_post_pre_with_rmsnorm_match_reference(tokens: int) -> None:
    device = require_sm120()
    hidden_size = 4096
    residual, x, fn, scale, bias = _make_inputs(
        tokens=tokens,
        hidden_size=hidden_size,
        seed=91_470 + tokens,
        device=device,
    )
    workspace = empty_mhc_workspace(
        num_tokens=tokens,
        hidden_size=hidden_size,
        device=device,
    )
    norm_gen = torch.Generator(device="cpu")
    norm_gen.manual_seed(91_471 + tokens)
    norm_weight = (
        torch.randn((hidden_size,), generator=norm_gen, dtype=torch.float32)
        .to(device)
        .to(torch.bfloat16)
        .contiguous()
    )
    _, prev_post, prev_comb = _mhc_pre_reference(
        residual,
        fn,
        scale,
        bias,
        rms_eps=1e-6,
        hc_eps=1e-6,
        sinkhorn_iters=20,
    )

    residual_cur, post, comb, y = b12x_mhc_post_pre(
        x,
        residual,
        prev_post.contiguous(),
        prev_comb.contiguous(),
        fn,
        scale,
        bias,
        rms_eps=1e-6,
        hc_eps=1e-6,
        sinkhorn_iters=20,
        workspace=workspace,
        norm_weight=norm_weight,
        norm_eps=1e-6,
    )
    torch.cuda.synchronize(device)

    residual_ref = _mhc_post_reference(x, residual, prev_post, prev_comb)
    y_raw_ref, post_ref, comb_ref = _mhc_pre_reference(
        residual_ref,
        fn,
        scale,
        bias,
        rms_eps=1e-6,
        hc_eps=1e-6,
        sinkhorn_iters=20,
    )
    y_ref = _rms_norm_reference(y_raw_ref, norm_weight, 1e-6)
    torch.testing.assert_close(residual_cur, residual_ref, rtol=0.0, atol=2e-2)
    torch.testing.assert_close(y, y_ref, rtol=0.0, atol=6e-3)
    torch.testing.assert_close(post, post_ref, rtol=2e-6, atol=1e-5)
    torch.testing.assert_close(comb, comb_ref, rtol=2e-6, atol=1e-5)


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
    y = torch.empty((tokens, hidden_size), dtype=torch.bfloat16, device=device)
    post = torch.empty((tokens, 4), dtype=torch.float32, device=device)
    comb = torch.empty((tokens, 4, 4), dtype=torch.float32, device=device)
    out = torch.empty_like(residual)

    def run() -> None:
        b12x_mhc_pre(
            residual,
            fn,
            scale,
            bias,
            rms_eps=1e-6,
            hc_eps=1e-6,
            sinkhorn_iters=20,
            y_out=y,
            post_out=post,
            comb_out=comb,
        )
        b12x_mhc_post(x, residual, post, comb, out=out)

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
    torch.testing.assert_close(post, post_ref, rtol=2e-6, atol=1e-5)
    torch.testing.assert_close(comb, comb_ref, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(out, out_ref, rtol=0.0, atol=2e-2)


def test_b12x_mhc_fused_pre_post_graph_capture() -> None:
    device = require_sm120()
    tokens = 2
    hidden_size = 4096
    residual, x, fn, scale, bias = _make_inputs(
        tokens=tokens,
        hidden_size=hidden_size,
        seed=91_440,
        device=device,
    )
    y = torch.empty((tokens, hidden_size), dtype=torch.bfloat16, device=device)
    post = torch.empty((tokens, 4), dtype=torch.float32, device=device)
    comb = torch.empty((tokens, 4, 4), dtype=torch.float32, device=device)
    out = torch.empty_like(residual)

    def run() -> None:
        b12x_mhc_pre_post(
            x,
            residual,
            fn,
            scale,
            bias,
            rms_eps=1e-6,
            hc_eps=1e-6,
            sinkhorn_iters=20,
            y_out=y,
            post_out=post,
            comb_out=comb,
            out=out,
        )

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
    out_ref = _mhc_post_reference(x, residual, post_ref, comb_ref)
    torch.testing.assert_close(y, y_ref, rtol=0.0, atol=2e-3)
    torch.testing.assert_close(post, post_ref, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(comb, comb_ref, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(out, out_ref, rtol=0.0, atol=2e-2)


def test_b12x_mhc_fused_post_pre_graph_capture() -> None:
    device = require_sm120()
    tokens = 2
    hidden_size = 4096
    residual, x, fn, scale, bias = _make_inputs(
        tokens=tokens,
        hidden_size=hidden_size,
        seed=91_460,
        device=device,
    )
    _, prev_post, prev_comb = _mhc_pre_reference(
        residual,
        fn,
        scale,
        bias,
        rms_eps=1e-6,
        hc_eps=1e-6,
        sinkhorn_iters=20,
    )
    prev_post_arg = prev_post.contiguous()
    prev_comb_arg = prev_comb.contiguous()
    residual_cur = torch.empty_like(residual)
    y = torch.empty((tokens, hidden_size), dtype=torch.bfloat16, device=device)
    post = torch.empty((tokens, 4), dtype=torch.float32, device=device)
    comb = torch.empty((tokens, 4, 4), dtype=torch.float32, device=device)

    def run() -> None:
        b12x_mhc_post_pre(
            x,
            residual,
            prev_post_arg,
            prev_comb_arg,
            fn,
            scale,
            bias,
            rms_eps=1e-6,
            hc_eps=1e-6,
            sinkhorn_iters=20,
            residual_out=residual_cur,
            y_out=y,
            post_out=post,
            comb_out=comb,
        )

    run()
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    torch.cuda.synchronize(device)

    residual_ref = _mhc_post_reference(x, residual, prev_post, prev_comb)
    y_ref, post_ref, comb_ref = _mhc_pre_reference(
        residual_ref,
        fn,
        scale,
        bias,
        rms_eps=1e-6,
        hc_eps=1e-6,
        sinkhorn_iters=20,
    )
    torch.testing.assert_close(residual_cur, residual_ref, rtol=0.0, atol=2e-2)
    torch.testing.assert_close(y, y_ref, rtol=0.0, atol=4e-3)
    torch.testing.assert_close(post, post_ref, rtol=2e-6, atol=1e-5)
    torch.testing.assert_close(comb, comb_ref, rtol=2e-6, atol=1e-5)


def test_b12x_mhc_post_match_reference_with_external_pre_outputs() -> None:
    device = require_sm120()
    tokens = 3
    hidden_size = 4096
    residual, x, fn, scale, bias = _make_inputs(
        tokens=tokens,
        hidden_size=hidden_size,
        seed=91_415,
        device=device,
    )
    _, post, comb = _mhc_pre_reference(
        residual,
        fn,
        scale,
        bias,
        rms_eps=1e-6,
        hc_eps=1e-6,
        sinkhorn_iters=20,
    )

    out = b12x_mhc_post(x, residual, post.contiguous(), comb.contiguous())
    torch.cuda.synchronize(device)

    out_ref = _mhc_post_reference(x, residual, post, comb)
    torch.testing.assert_close(out, out_ref, rtol=0.0, atol=2e-2)


def test_b12x_mhc_pre_rejects_non_cute_split_k() -> None:
    device = require_sm120()
    residual, _, fn, scale, bias = _make_inputs(
        tokens=1,
        hidden_size=4096,
        seed=91_420,
        device=device,
    )
    with pytest.raises(ValueError, match="requires split_k=64"):
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

    with pytest.raises(ValueError, match="requires split_k=64"):
        b12x_mhc_pre_post(
            torch.empty((1, 4096), dtype=torch.bfloat16, device=device),
            residual,
            fn,
            scale,
            bias,
            rms_eps=1e-6,
            hc_eps=1e-6,
            sinkhorn_iters=20,
            split_k=1,
        )

    _, prev_post, prev_comb = _mhc_pre_reference(
        residual,
        fn,
        scale,
        bias,
        rms_eps=1e-6,
        hc_eps=1e-6,
        sinkhorn_iters=20,
    )
    with pytest.raises(ValueError, match="requires split_k=64"):
        b12x_mhc_post_pre(
            torch.empty((1, 4096), dtype=torch.bfloat16, device=device),
            residual,
            prev_post.contiguous(),
            prev_comb.contiguous(),
            fn,
            scale,
            bias,
            rms_eps=1e-6,
            hc_eps=1e-6,
            sinkhorn_iters=20,
            split_k=1,
        )
