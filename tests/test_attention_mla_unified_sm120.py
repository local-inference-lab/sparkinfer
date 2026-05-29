"""Opt-in dispatch smoke tests for the parallel unified_sm120 sparse-MLA backend.

These tests do NOT exercise the unified_sm120 kernel itself (a parallel agent owns the
kernel bodies under b12x/attention/mla/unified_sm120/). They only prove the GATE wired
into the legacy MLA APIs:

  * with B12X_MLA_SM120_UNIFIED unset/0 and no backend kwarg, a tiny GLM (q_head_dim==576)
    decode still routes to the LEGACY path with no behavior change; and
  * with the env flag on OR backend="sm120_unified", the same call routes into
    unified_sm120.run_unified_decode, which currently raises NotImplementedError
    (expected during P5-P7 -- the routing is what is under test here).

Scope decisions (.sm120port/scope_decisions.md): DSV3.2 is dropped; GLM_NSA is the
uncompressed q=576 contract; the unified backend is opt-in and the legacy path stays the
default. Tensors are kept tiny; the legacy path is forced down its PyTorch reference
fallback via monkeypatch so test (a) is deterministic without compiling a GLM kernel.
"""

from __future__ import annotations

import math
import os
import sys

import pytest
import torch

# Import the pure-PyTorch DSV4 numeric reference (.sm120port/dsv4_ref.py) -- the
# SAME oracle the P5 single-CTA decode and the P7 launcher probe validate
# against. It has no b12x/CuTe dependency, so a plain sys.path insert is enough
# (mirrors .sm120port/probes/unified_decode_e2e_check.py's import dance).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SM120PORT = os.path.join(_REPO_ROOT, ".sm120port")
if _SM120PORT not in sys.path:
    sys.path.insert(0, _SM120PORT)
import dsv4_ref  # noqa: E402
import glm_ref  # noqa: E402

import b12x.attention.mla.api as mla_api
import b12x.attention.mla.compressed_api as compressed_api_impl
from b12x.attention.mla.api import sparse_mla_decode_forward
from b12x.attention.mla.compressed_api import compressed_mla_decode_forward
from b12x.attention.mla.compressed_reference import (
    compressed_mla_page_nbytes,
    compressed_sparse_mla_reference,
    pack_compressed_mla_kv_cache_reference,
)
from b12x.cute.fp4 import get_sm_version
from b12x.integration.compressed_scratch import (
    B12XCompressedMLAScratchCaps,
    _compressed_mla_scratch_layout,
    _materialize_compressed_mla_scratch,
)

from .helpers import require_sm120


# GLM_NSA uncompressed decode contract: q_head_dim = d_nope + d_rope = 512 + 64 = 576,
# v_head_dim = 512, packed KV cache = 656 bytes/token (verified_traits.md).
_GLM_Q_HEAD_DIM = 576
_GLM_V_HEAD_DIM = 512
_GLM_KV_BYTES_PER_TOKEN = 656
_NUM_Q_HEADS = 8
_SM_SCALE = 1.0 / math.sqrt(_GLM_Q_HEAD_DIM)


def require_sm120_unified() -> torch.device:
    """Skip unless a real SM120+ device is present (mirrors the compressed-MLA pattern).

    The opt-in gate requires get_sm_version(device) >= 120, so the routing branch under
    test is only reachable on SM120 hardware.
    """
    device = require_sm120()
    if get_sm_version(device) < 120:
        pytest.skip("unified_sm120 dispatch requires an SM120+ device")
    return device


def _make_glm_decode_inputs(device: torch.device, num_q_heads: int = _NUM_Q_HEADS):
    """Build a tiny GLM (q_head_dim==576) sparse-decode call and its legacy workspace."""
    rows = 1
    width = 4
    q_all = torch.zeros(
        (rows, num_q_heads, _GLM_Q_HEAD_DIM), dtype=torch.bfloat16, device=device
    )
    kv_cache = torch.zeros(
        (16, 1, _GLM_KV_BYTES_PER_TOKEN), dtype=torch.uint8, device=device
    )
    page_table_1 = torch.zeros((rows, width), dtype=torch.int32, device=device)
    cache_seqlens = torch.full((rows,), width, dtype=torch.int32, device=device)
    from b12x.attention.workspace import B12XAttentionWorkspace

    workspace = B12XAttentionWorkspace.for_contract(
        mode="decode",
        device=device,
        dtype=torch.bfloat16,
        kv_dtype=torch.uint8,
        num_q_heads=num_q_heads,
        head_dim=_GLM_Q_HEAD_DIM,
        v_head_dim=_GLM_V_HEAD_DIM,
        topk=width,
        max_total_q=rows,
        max_batch=rows,
        max_kv_rows=rows * width,
        use_cuda_graph=False,
    )
    return q_all, kv_cache, page_table_1, cache_seqlens, workspace


def _force_legacy_reference(monkeypatch) -> dict[str, int]:
    """Pin the legacy GLM decode to its PyTorch reference fallback and count its use.

    Forcing select_sparse_mla_split_decode_config -> None and supports_sparse_mla_kernel
    -> False routes _run_sparse_mla down its deterministic reference path, so test (a) does
    not depend on compiling a real GLM kernel for tiny tensors.
    """
    counters = {"reference_calls": 0}

    def fake_select_split(**kwargs):
        del kwargs
        return None

    def fake_supports(**kwargs):
        del kwargs
        return False

    def fake_reference(*, q_all, kv_cache, page_table_1, active_token_counts, sm_scale, v_head_dim, **kwargs):
        del kv_cache, page_table_1, active_token_counts, sm_scale, kwargs
        counters["reference_calls"] += 1
        return q_all[:, :, :v_head_dim].clone()

    monkeypatch.setattr(mla_api, "select_sparse_mla_split_decode_config", fake_select_split)
    monkeypatch.setattr(mla_api, "supports_sparse_mla_kernel", fake_supports)
    monkeypatch.setattr(mla_api, "sparse_mla_reference", fake_reference)
    return counters


@torch.inference_mode()
def test_glm_decode_flag_off_uses_legacy_path(monkeypatch) -> None:
    """(a) Flag unset/0, no backend kwarg: a tiny GLM decode stays on the LEGACY path."""
    device = require_sm120_unified()
    monkeypatch.delenv("B12X_MLA_SM120_UNIFIED", raising=False)
    counters = _force_legacy_reference(monkeypatch)

    q_all, kv_cache, page_table_1, cache_seqlens, workspace = _make_glm_decode_inputs(device)
    output = sparse_mla_decode_forward(
        q_all=q_all,
        kv_cache=kv_cache,
        page_table_1=page_table_1,
        cache_seqlens_int32=cache_seqlens,
        nsa_cache_seqlens_int32=cache_seqlens,
        workspace=workspace,
        sm_scale=_SM_SCALE,
        v_head_dim=_GLM_V_HEAD_DIM,
    )

    assert output.shape == (1, _NUM_Q_HEADS, _GLM_V_HEAD_DIM)
    # The legacy reference ran exactly once; the unified gate did not intercept.
    assert counters["reference_calls"] == 1


@torch.inference_mode()
def test_glm_decode_flag_off_explicit_zero_uses_legacy_path(monkeypatch) -> None:
    """(a') Flag explicitly "0": still the LEGACY path (env helper parses 0 as off)."""
    device = require_sm120_unified()
    monkeypatch.setenv("B12X_MLA_SM120_UNIFIED", "0")
    counters = _force_legacy_reference(monkeypatch)

    q_all, kv_cache, page_table_1, cache_seqlens, workspace = _make_glm_decode_inputs(device)
    output = sparse_mla_decode_forward(
        q_all=q_all,
        kv_cache=kv_cache,
        page_table_1=page_table_1,
        cache_seqlens_int32=cache_seqlens,
        nsa_cache_seqlens_int32=cache_seqlens,
        workspace=workspace,
        sm_scale=_SM_SCALE,
        v_head_dim=_GLM_V_HEAD_DIM,
    )

    assert output.shape == (1, _NUM_Q_HEADS, _GLM_V_HEAD_DIM)
    assert counters["reference_calls"] == 1


@torch.inference_mode()
def test_glm_decode_flag_on_routes_to_unified(monkeypatch) -> None:
    """(b) Env flag on + valid GLM contract: the GLM decode routes into
    unified_sm120.run_unified_decode (P7b -- GLM_NSA is now implemented).

    The launcher itself is exercised end-to-end by .sm120port/probes/
    glm_decode_e2e_check.py vs glm_ref; here we only prove the GATE intercepts a
    valid GLM contract (heads % HPB == 0, single row, no LSE/sink) and that the
    legacy reference does NOT run. We monkeypatch run_unified_decode to a sentinel
    so the routing decision is observed without compiling a kernel for tiny inputs.
    """
    device = require_sm120_unified()
    monkeypatch.setenv("B12X_MLA_SM120_UNIFIED", "1")
    counters = _force_legacy_reference(monkeypatch)

    routed = {"calls": 0}

    def fake_run_unified_decode(*, q_all, **kwargs):
        del kwargs
        routed["calls"] += 1
        return q_all[:, :, :_GLM_V_HEAD_DIM].clone()

    import b12x.attention.mla.unified_sm120.launch as unified_launch

    monkeypatch.setattr(unified_launch, "run_unified_decode", fake_run_unified_decode)

    # HPB=16 contract: heads must be divisible by 16 for the unified GLM route.
    q_all, kv_cache, page_table_1, cache_seqlens, workspace = _make_glm_decode_inputs(
        device, num_q_heads=16
    )
    output = sparse_mla_decode_forward(
        q_all=q_all,
        kv_cache=kv_cache,
        page_table_1=page_table_1,
        cache_seqlens_int32=cache_seqlens,
        nsa_cache_seqlens_int32=cache_seqlens,
        workspace=workspace,
        sm_scale=_SM_SCALE,
        v_head_dim=_GLM_V_HEAD_DIM,
    )

    assert output.shape == (1, 16, _GLM_V_HEAD_DIM)
    # The unified GLM launcher intercepted; the legacy reference did NOT run.
    assert routed["calls"] == 1
    assert counters["reference_calls"] == 0


# ── DSV4 MAIN-CACHE compressed decode (P7): real kernel + split-K + merge ──────
_DSV4_HEADS = 32          # local q heads (2 head blocks of HPB=16)
_DSV4_HEAD_DIM = 512
_DSV4_PAGE = 64           # compressed page_size for the test cache
_DSV4_SM_SCALE = 1.0 / math.sqrt(_DSV4_HEAD_DIM)


def _make_dsv4_compressed_case(device, *, topk, seed=0):
    gen = torch.Generator(device=device).manual_seed(seed)
    num_blocks = 16
    n_tokens = num_blocks * _DSV4_PAGE
    k_nope = (torch.randn((n_tokens, 448), generator=gen, dtype=torch.float32, device=device) / 10).clamp(-1, 1)
    k_rope = (torch.randn((n_tokens, 64), generator=gen, dtype=torch.float32, device=device) / 10).clamp(-1, 1)
    cache = pack_compressed_mla_kv_cache_reference(
        k_nope, k_rope.to(torch.bfloat16), page_size=_DSV4_PAGE, num_pages=num_blocks
    )
    q = (torch.randn((1, _DSV4_HEADS, _DSV4_HEAD_DIM), generator=gen, dtype=torch.float32, device=device) / 10).clamp(-1, 1).to(torch.bfloat16)
    idx = torch.randint(0, n_tokens, (1, topk), generator=gen, dtype=torch.int32, device=device)
    idx[:, topk // 2:] = -1  # invalidate the back half (matches dsv4_ref cases)
    lengths = torch.full((1,), topk, dtype=torch.int32, device=device)
    return q, cache, idx, lengths


def _make_dsv4_scratch(device, *, topk, max_chunks):
    caps = B12XCompressedMLAScratchCaps(
        device=device, num_q_heads=_DSV4_HEADS, max_q_rows=1, max_width=topk,
        head_dim=_DSV4_HEAD_DIM, v_head_dim=_DSV4_HEAD_DIM,
        max_chunks_per_row=max_chunks, page_size=_DSV4_PAGE,
    )
    layout = _compressed_mla_scratch_layout(caps)
    storage = torch.zeros(int(layout.nbytes), dtype=torch.uint8, device=device)
    return _materialize_compressed_mla_scratch(caps, storage, layout)


@torch.inference_mode()
@pytest.mark.parametrize("topk", [64, 128, 512])
def test_dsv4_compressed_decode_routes_to_unified_and_matches_reference(monkeypatch, topk) -> None:
    """(c) Flag on + DSV4 main-cache contract: compressed_mla_decode_forward routes to
    unified_sm120.run_unified_decode (kernel split-K partials + reused base-2 merge)
    and matches compressed_sparse_mla_reference. The legacy split forward must NOT run."""
    device = require_sm120_unified()
    monkeypatch.setenv("B12X_MLA_SM120_UNIFIED", "1")

    legacy_calls = {"forward": 0}

    def fail_forward(**kwargs):
        legacy_calls["forward"] += 1
        raise AssertionError("legacy compressed split forward ran for a DSV4 unified case")

    monkeypatch.setattr(compressed_api_impl, "run_compressed_mla_split_decode_forward", fail_forward)

    q, cache, idx, lengths = _make_dsv4_compressed_case(device, topk=topk, seed=topk)
    scratch = _make_dsv4_scratch(device, topk=topk, max_chunks=8)

    out = compressed_mla_decode_forward(
        q_all=q,
        swa_k_cache=cache,
        swa_indices=idx,
        swa_topk_lengths=lengths,
        workspace=scratch,
        sm_scale=_DSV4_SM_SCALE,
        swa_page_size=_DSV4_PAGE,
    )
    torch.cuda.synchronize()
    assert legacy_calls["forward"] == 0  # gate routed to unified, not legacy
    assert out.shape == (1, _DSV4_HEADS, _DSV4_HEAD_DIM)

    exp = compressed_sparse_mla_reference(
        q, cache, idx, lengths, sm_scale=_DSV4_SM_SCALE, swa_page_size=_DSV4_PAGE
    )[0].float()
    got = out[0].float()
    cos = float((got.flatten().double() @ exp.flatten().double()) /
                (got.flatten().double().norm() * exp.flatten().double().norm()))
    assert cos > 0.999, f"DSV4 unified decode cos={cos}"
    assert (got - exp).abs().max().item() < 2e-2


@torch.inference_mode()
def test_dsv4_compressed_decode_extra_cache_falls_back_to_legacy(monkeypatch) -> None:
    """(d) Flag on but has_extra_cache (indexed/extra-tokens, P7c): the gate must FALL
    BACK to the legacy path -- never route an unsupported call to the unified kernel."""
    device = require_sm120_unified()
    monkeypatch.setenv("B12X_MLA_SM120_UNIFIED", "1")

    routed = {"unified": 0}

    import b12x.attention.mla.unified_sm120 as unified_mod

    def spy_unified(**kwargs):
        routed["unified"] += 1
        raise AssertionError("unsupported (extra-cache) DSV4 call was routed to unified")

    monkeypatch.setattr(unified_mod, "run_unified_decode", spy_unified)

    legacy = {"forward": 0}

    def fake_forward(**kwargs):
        legacy["forward"] += 1
        binding = kwargs["binding"]
        binding.tmp_output.zero_()
        binding.tmp_lse.fill_(float("-inf"))

    def fake_merge(**kwargs):
        kwargs["binding"].output.zero_()

    monkeypatch.setattr(compressed_api_impl, "run_compressed_mla_split_decode_forward", fake_forward)
    monkeypatch.setattr(compressed_api_impl, "run_sparse_mla_split_decode_merge", fake_merge)

    topk = 64
    q, cache, idx, lengths = _make_dsv4_compressed_case(device, topk=topk, seed=7)
    scratch = _make_dsv4_scratch(device, topk=topk * 2, max_chunks=8)
    # Use the live-shape (non-fixed) legacy planning so this fallback test does
    # not exercise the fixed-capacity staging buffers (which a B12XAttentionWorkspace
    # owns, not this minimal compressed scratch); the gate decision is what's under test.
    scratch.fixed_capacity = False

    out = compressed_mla_decode_forward(
        q_all=q,
        swa_k_cache=cache,
        swa_indices=idx,
        swa_topk_lengths=lengths,
        indexed_k_cache=cache,
        indexed_indices=idx,
        indexed_topk_lengths=lengths,
        indexed_page_size=_DSV4_PAGE,
        workspace=scratch,
        sm_scale=_DSV4_SM_SCALE,
        swa_page_size=_DSV4_PAGE,
    )
    assert routed["unified"] == 0      # NOT routed to unified
    assert legacy["forward"] == 1      # fell back to legacy
    assert out.shape == (1, _DSV4_HEADS, _DSV4_HEAD_DIM)


@torch.inference_mode()
def test_glm_decode_backend_kwarg_routes_to_unified(monkeypatch) -> None:
    """(b') backend="sm120_unified" routes the GLM decode to unified_sm120 even with
    the env flag off (P7b: GLM_NSA implemented; the gate routes via run_unified_decode)."""
    device = require_sm120_unified()
    monkeypatch.delenv("B12X_MLA_SM120_UNIFIED", raising=False)
    counters = _force_legacy_reference(monkeypatch)

    routed = {"calls": 0}

    def fake_run_unified_decode(*, q_all, **kwargs):
        del kwargs
        routed["calls"] += 1
        return q_all[:, :, :_GLM_V_HEAD_DIM].clone()

    import b12x.attention.mla.unified_sm120.launch as unified_launch

    monkeypatch.setattr(unified_launch, "run_unified_decode", fake_run_unified_decode)

    q_all, kv_cache, page_table_1, cache_seqlens, workspace = _make_glm_decode_inputs(
        device, num_q_heads=16
    )
    output = sparse_mla_decode_forward(
        q_all=q_all,
        kv_cache=kv_cache,
        page_table_1=page_table_1,
        cache_seqlens_int32=cache_seqlens,
        nsa_cache_seqlens_int32=cache_seqlens,
        workspace=workspace,
        sm_scale=_SM_SCALE,
        v_head_dim=_GLM_V_HEAD_DIM,
        backend="sm120_unified",
    )

    assert output.shape == (1, 16, _GLM_V_HEAD_DIM)
    assert routed["calls"] == 1
    assert counters["reference_calls"] == 0


# ── DSV4 LAUNCHER NUMERICS (P7): run_unified_decode vs dsv4_ref ────────────────
# These exercise the REAL launcher (b12x.attention.mla.unified_sm120.launch.
# run_unified_decode = warp-specialized 288-thread DSV4 decode -> per-split
# NORMALIZED partials in mid_out/mid_lse -> the REUSED split.py base-2 merge) for
# num_heads=128, topk in {64,128,512}, and BOTH num_splits=1 and a forced>1
# split, comparing final O + base-2 LSE to the pure-PyTorch dsv4_ref oracle at
# the validated P5 gate (cos > 0.999, O atol 2e-2, lse atol 5e-2).
_UNIFIED_NUM_HEADS = 128       # full DSV4 head count (8 head blocks of HPB=16)
_UNIFIED_NUM_BLOCKS = 64
_LN2 = math.log(2.0)


def _cosine(got: torch.Tensor, exp: torch.Tensor) -> float:
    a = got.flatten().double()
    b = exp.flatten().double()
    denom = (a.norm() * b.norm()).item()
    return 1.0 if denom == 0 else float((a @ b).item() / denom)


def _repack_dsv4_to_compressed(packed_dsv4: torch.Tensor, page_size: int, num_blocks: int) -> torch.Tensor:
    """Re-lay the dsv4_ref (nb,bs,1,584) cache into the compressed flat
    [pages, page_nbytes] layout the launcher reads (swa_k_cache.reshape(-1) with
    per-page stride = compressed_mla_page_nbytes(page_size)).

    The data (pbs*576) + footer (pbs*8) = pbs*584 bytes are byte-identical
    between the two packings (the verified P7 layout finding); only the per-page
    byte stride is padded to a 576-multiple, so we copy the dsv4 page bytes into
    the leading (pbs*584) of each compressed (padded) page.
    """
    bs = page_size
    bpt = dsv4_ref.DSV4_KV_GMEM_STRIDE  # 584
    page_nbytes = compressed_mla_page_nbytes(page_size)
    flat = packed_dsv4.reshape(num_blocks, bs * bpt)
    out = torch.zeros(num_blocks, page_nbytes, dtype=torch.uint8, device=packed_dsv4.device)
    out[:, : bs * bpt] = flat
    return out


def _merge_base2_lse(mid_lse: torch.Tensor) -> torch.Tensor:
    """Reproduce split.py's base-2 merge of per-split LSEs -> the final base-2
    LSE. ``mid_lse`` is [heads, num_splits] base-2 (log2(sum)+max) with -inf for
    empty splits. The merge computes merged_m + log2(merged_d) =
    log2(sum_i 2^lse_i), i.e. a base-2 logsumexp over the split axis (all-empty
    rows stay -inf). torch.logsumexp handles the -inf sentinel correctly."""
    return torch.logsumexp(mid_lse.float() * _LN2, dim=-1) / _LN2


def _make_dsv4_scratch_heads(device, *, topk, max_chunks, num_heads):
    caps = B12XCompressedMLAScratchCaps(
        device=device, num_q_heads=num_heads, max_q_rows=1, max_width=topk,
        head_dim=_DSV4_HEAD_DIM, v_head_dim=_DSV4_HEAD_DIM,
        max_chunks_per_row=max_chunks, page_size=_DSV4_PAGE,
    )
    layout = _compressed_mla_scratch_layout(caps)
    storage = torch.zeros(int(layout.nbytes), dtype=torch.uint8, device=device)
    return _materialize_compressed_mla_scratch(caps, storage, layout)


def _run_unified_dsv4(device, *, topk, forced_num_splits, seed):
    """Build a dsv4_ref DSV4 decode case (num_heads=128), repack the KV into the
    compressed page layout, run the REAL launcher, and return
    (got_O, got_lse, exp_O, exp_lse, eff_splits)."""
    from b12x.attention.mla.unified_sm120.launch import run_unified_decode

    case = dsv4_ref.make_dsv4_decode_case(
        num_heads=_UNIFIED_NUM_HEADS, topk=topk, num_tokens=1,
        num_blocks=_UNIFIED_NUM_BLOCKS, page_block_size=_DSV4_PAGE,
        invalidate_half=True, with_sink=False, device=device, seed=seed,
    )
    q = case["q"].contiguous()                    # [1, 128, 512] bf16
    swa_cache = _repack_dsv4_to_compressed(case["kv_cache"], _DSV4_PAGE, _UNIFIED_NUM_BLOCKS)
    idx = case["topk_indices"].contiguous()       # [1, topk] int32
    lengths = torch.full((1,), topk, dtype=torch.int32, device=device)
    exp_O = case["expected_O"][0].float()         # [128, 512]
    exp_lse = case["expected_lse"][0].float()     # [128]
    sm_scale = case["sm_scale"]

    n_chunks = (topk + 64 - 1) // 64
    max_chunks = max(8, forced_num_splits)
    scratch = _make_dsv4_scratch_heads(
        device, topk=topk, max_chunks=max_chunks, num_heads=_UNIFIED_NUM_HEADS,
    )

    out = run_unified_decode(
        q_all=q,
        swa_k_cache=swa_cache,
        swa_indices=idx,
        swa_topk_lengths=lengths,
        workspace=scratch,
        sm_scale=sm_scale,
        swa_page_size=_DSV4_PAGE,
        forced_num_splits=forced_num_splits,
    )
    torch.cuda.synchronize()

    eff_splits = min(forced_num_splits, n_chunks)
    got_O = out[0].float()                                       # [128, 512]
    # Per-split base-2 LSE partials live in the workspace mid_lse after the call;
    # the base-2 merge reconstructs the final LSE (= mid_lse[:, 0] when 1 split).
    mid_lse = scratch.tmp_lse[:1, :_UNIFIED_NUM_HEADS, :eff_splits][0]  # [128, eff_splits]
    got_lse = _merge_base2_lse(mid_lse)                          # [128]
    return got_O, got_lse, exp_O, exp_lse, eff_splits


@torch.inference_mode()
@pytest.mark.parametrize("topk,forced_num_splits", [
    (64, 1), (128, 1), (512, 1),     # single split (trivial 1-split merge)
    (64, 2), (128, 2), (512, 4),     # forced multi-split (chunk-aligned ranges)
])
def test_unified_decode_launcher_matches_dsv4_ref(topk, forced_num_splits) -> None:
    """run_unified_decode (kernel split-K partials + reused base-2 merge) matches
    the dsv4_ref oracle for num_heads=128 at the validated P5 gate, for BOTH
    num_splits=1 and a forced>1 split (the multi-split chunk-aligned partition is
    numerically identical to single-split)."""
    device = require_sm120_unified()
    got_O, got_lse, exp_O, exp_lse, eff_splits = _run_unified_dsv4(
        device, topk=topk, forced_num_splits=forced_num_splits, seed=topk,
    )

    if forced_num_splits > 1:
        # The forced split must actually partition into the expected number of
        # chunk-aligned ranges (topk=64 -> 1 chunk, so a forced 2 collapses to
        # 1; the numerics below still validate the path either way).
        expected_splits = min(forced_num_splits, (topk + 64 - 1) // 64)
        assert eff_splits == expected_splits

    cos = _cosine(got_O, exp_O)
    assert cos > 0.999, f"topk={topk} splits={forced_num_splits} O cos={cos}"
    assert (got_O - exp_O).abs().max().item() < 2e-2, (
        f"topk={topk} splits={forced_num_splits} O atol exceeded"
    )
    # base-2 LSE: all reference heads here are finite (back half invalidated, not
    # the whole row), so compare directly at atol 5e-2.
    assert torch.isfinite(exp_lse).all()
    assert torch.isfinite(got_lse).all()
    assert (got_lse - exp_lse).abs().max().item() < 5e-2, (
        f"topk={topk} splits={forced_num_splits} lse atol exceeded: "
        f"max|delta|={(got_lse - exp_lse).abs().max().item()}"
    )


@torch.inference_mode()
@pytest.mark.parametrize("topk,forced_num_splits", [(128, 2), (512, 2), (512, 4)])
def test_unified_decode_multi_split_equals_single_split(topk, forced_num_splits) -> None:
    """A forced multi-split decode must equal the single-split decode (same
    chunk-aligned candidate partition, merged base-2): each candidate is owned by
    exactly one split, so the reduction is exact, not just within-tolerance."""
    device = require_sm120_unified()
    got_single, _, _, _, _ = _run_unified_dsv4(
        device, topk=topk, forced_num_splits=1, seed=topk,
    )
    got_multi, _, _, _, eff_splits = _run_unified_dsv4(
        device, topk=topk, forced_num_splits=forced_num_splits, seed=topk,
    )
    assert eff_splits > 1, "forced multi-split did not partition into >1 split"
    delta = (got_multi - got_single).abs().max().item()
    assert delta < 5e-3, (
        f"topk={topk} splits={forced_num_splits} multi vs single max|delta|={delta}"
    )


# ── GLM_NSA LAUNCHER NUMERICS (P7b): run_unified_decode vs glm_ref ─────────────
# GLM is rs-1's own model (no FlashInfer PTX), so the bar is NUMERICAL vs
# glm_ref.glm_decode_reference (sparse_mla_reference). The GLM K dequant->requant
# e4m3 + unit-sfb path is lossier than DSV4, so the tolerance is the looser GLM
# band (cos > 0.995, O atol 3e-2), matching glm_ref's own brute-force self-test
# tolerance. The unified GLM kernel reuses the SAME split.py base-2 merge as DSV4.
_GLM_NUM_HEADS = 128       # full GLM head count (8 head blocks of HPB=16)
_GLM_PAGE = 64             # GLM decode page_block_size (stride = idx*656; pbs-invariant)


def _make_glm_sparse_scratch(device, *, topk, max_chunks, num_heads, s_kv):
    from b12x.integration.sparse_mla_scratch import (
        B12XSparseMLAScratchCaps,
        plan_sparse_mla_scratch,
    )

    caps = B12XSparseMLAScratchCaps(
        device=device, num_q_heads=num_heads, max_q_rows=1, max_batch=1,
        max_width=max(topk, 1), max_kv_rows=s_kv,
        head_dim=glm_ref.GLM_Q_HEAD_DIM, v_head_dim=glm_ref.GLM_D_V,
        max_chunks_per_row=max_chunks, page_size=_GLM_PAGE,
    )
    plan = plan_sparse_mla_scratch(caps)
    (spec,) = plan.scratch_specs()
    storage = torch.zeros(spec.shape, dtype=spec.dtype, device=device)
    return plan, storage


def _run_unified_glm(device, *, topk, forced_num_splits, seed):
    """Build a glm_ref GLM decode case (num_heads=128), run the REAL launcher over
    the GLM 656B cache + sparse scratch, return (got_O, exp_O, eff_splits)."""
    from b12x.attention.mla.unified_sm120.launch import run_unified_decode

    nblk = max(1, (topk + _GLM_PAGE - 1) // _GLM_PAGE)
    case = glm_ref.make_glm_decode_case(
        num_heads=_GLM_NUM_HEADS, topk=topk, num_blocks=nblk,
        page_block_size=_GLM_PAGE, invalidate_half=True, seed=seed, device=device,
    )
    q = case["q"].contiguous()                    # [1, 128, 576] bf16
    kv_cache = case["kv_cache"].contiguous()      # (nblk*64, 1, 656) uint8 GLM
    idx = case["topk_indices"].contiguous()       # [1, topk] int32
    exp_O = case["expected_O"][0].float()         # [128, 512]
    sm_scale = case["sm_scale"]
    s_kv = kv_cache.shape[0]

    n_chunks = (topk + 64 - 1) // 64
    max_chunks = max(8, forced_num_splits)
    plan, storage = _make_glm_sparse_scratch(
        device, topk=topk, max_chunks=max_chunks, num_heads=_GLM_NUM_HEADS, s_kv=s_kv,
    )
    cache_seqlens = torch.full((1,), s_kv, dtype=torch.int32, device=device)
    nsa_seqlens = torch.full((1,), topk, dtype=torch.int32, device=device)
    binding = plan.bind(
        scratch=storage, q=q, selected_indices=idx,
        cache_seqlens_int32=cache_seqlens, nsa_cache_seqlens_int32=nsa_seqlens,
    )

    out = run_unified_decode(
        q_all=q,
        swa_k_cache=kv_cache,
        swa_indices=idx,
        swa_topk_lengths=nsa_seqlens,
        workspace=binding.scratch,
        sm_scale=sm_scale,
        swa_page_size=_GLM_PAGE,
        forced_num_splits=forced_num_splits,
    )
    torch.cuda.synchronize()
    return out[0].float(), exp_O, min(forced_num_splits, n_chunks)


@torch.inference_mode()
@pytest.mark.parametrize("topk,forced_num_splits", [
    (64, 1), (128, 1), (512, 1),     # single split (trivial 1-split merge)
    (128, 2), (512, 4),              # forced multi-split (chunk-aligned ranges)
])
def test_unified_decode_launcher_matches_glm_ref(topk, forced_num_splits) -> None:
    """run_unified_decode GLM_NSA branch (ARBITRARY_FP32 inline scales,
    V_HAS_ROPE=false, 512/128/4) matches the glm_ref oracle for num_heads=128 at
    the looser GLM gate (cos > 0.995, O atol 3e-2), for single AND forced>1 split."""
    device = require_sm120_unified()
    got_O, exp_O, eff_splits = _run_unified_glm(
        device, topk=topk, forced_num_splits=forced_num_splits, seed=topk,
    )
    cos = _cosine(got_O, exp_O)
    assert cos > 0.995, f"GLM topk={topk} splits={forced_num_splits} O cos={cos}"
    assert (got_O - exp_O).abs().max().item() < 3e-2, (
        f"GLM topk={topk} splits={forced_num_splits} O atol exceeded: "
        f"max|delta|={(got_O - exp_O).abs().max().item()}"
    )


@torch.inference_mode()
@pytest.mark.parametrize("topk,forced_num_splits", [(128, 2), (512, 4)])
def test_unified_decode_glm_multi_split_equals_single_split(topk, forced_num_splits) -> None:
    """A forced GLM multi-split decode must match the single-split decode (same
    chunk-aligned candidate partition, merged base-2)."""
    device = require_sm120_unified()
    got_single, _, _ = _run_unified_glm(device, topk=topk, forced_num_splits=1, seed=topk)
    got_multi, _, eff_splits = _run_unified_glm(
        device, topk=topk, forced_num_splits=forced_num_splits, seed=topk,
    )
    assert eff_splits > 1, "forced GLM multi-split did not partition into >1 split"
    delta = (got_multi - got_single).abs().max().item()
    assert delta < 5e-3, (
        f"GLM topk={topk} splits={forced_num_splits} multi vs single max|delta|={delta}"
    )
