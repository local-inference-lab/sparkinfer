#!/usr/bin/env python3
"""Graph-captured latency + KV-bandwidth comparison for SM120 sparse-MLA.

GOAL (rs-1 SM120 port, benchmark phase): show the unified CuTeDSL backend is
PERFORMANT, not merely correct -- the payoff of PTX parity. Three contracts:

  * DSV4 decode  : legacy compressed_mla_decode_forward (flag OFF)
                   vs unified run_unified_decode (flag ON), num_heads=128,
                   topk in {512,1024,2048}, q_head_dim=512.  Shapes mirror
                   benchmarks/benchmark_compressed_mla.py.
  * GLM  decode  : legacy sparse_mla_decode_forward (flag OFF)
                   vs unified run_unified_decode (flag ON), q_head_dim=576,
                   656B/token KV.  Shapes mirror benchmarks/benchmark_mla.py
                   (GLM-5.1 contract; heads % 16 == 0 for the HPB=16 route).
  * DSV4 prefill : unified run_unified_prefill (single-pass) vs the closest
                   legacy comparable (legacy compressed split-decode over the
                   same T prefill rows -- there is NO dedicated legacy compressed
                   prefill kernel, so this is the nearest apples-to-apples
                   legacy compressed path; noted explicitly).

ANTI-FALLBACK PROOF.  The unified dispatch FALLS BACK to legacy for unsupported
features, so a silent fallback would make unified==legacy (fake parity).  We do
NOT trust the env flag.  We install TWO spies:

  (1) launch counter: wrap sparkinfer._lib.compiler.launch and count every launch
      whose compile_spec.kernel_id contains "sm120" (i.e. the REAL
      UnifiedDecodeKernel / UnifiedPrefillKernel cubin compile+launch).  CUDA
      graphs capture these launches during warmup+capture; replay re-issues the
      captured kernel WITHOUT re-entering Python, so a non-zero launch-counter
      delta over warmup+capture PROVES the unified cubin is what's in the graph.
  (2) dispatch wrapper: pass-through counters on run_unified_decode /
      run_unified_prefill at every import site, proving the gate ROUTED to the
      unified entrypoint (vs falling through to legacy).

Per config we report legacy_us, unified_us, speedup (legacy/unified), and
unified_actually_ran (the AND of the two spies firing for the unified run).
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import math
import os
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from benchmarks.common import (
    bench_cuda_graph,
    capture_cuda_graph,
    make_l2_flush_fn,
    require_sm120,
    resolve_l2_flush_bytes,
)

from sparkinfer.attention._shared.workspace import SPARKINFERAttentionWorkspace
from sparkinfer.attention._shared.mla.compressed_reference import (
    COMPRESSED_MLA_DSV4_PAGE_SIZE,
    COMPRESSED_MLA_HEAD_DIM,
    COMPRESSED_MLA_NOPE_DIM,
    COMPRESSED_MLA_ROPE_DIM,
    compressed_mla_page_nbytes,
    pack_compressed_mla_kv_cache_reference,
)
from sparkinfer.attention._shared.mla.reference import pack_mla_kv_cache_reference
from sparkinfer.attention._shared.mla.api import clear_mla_caches, sparse_mla_decode_forward
from sparkinfer.attention._shared.mla.compressed_api import compressed_mla_decode_forward

_MLA_NOPE_DIM = 512   # GLM kv_lora_rank (nope) dim
_MLA_ROPE_DIM = 64    # GLM qk_rope_head_dim
_GLM_Q_HEAD_DIM = _MLA_NOPE_DIM + _MLA_ROPE_DIM      # 576
_GLM_V_HEAD_DIM = _MLA_NOPE_DIM                       # 512
_GLM_KV_BYTES_PER_TOKEN = 656
_DSV4_HEAD_DIM = COMPRESSED_MLA_HEAD_DIM              # 512
_DSV4_PREFILL_PAGE = 64    # the page_block_size the unified prefill kernel uses

_UNIFIED_ENV = "SPARKINFER_MLA_SM120_ROUTE_REMOVED"


# --------------------------------------------------------------------------- #
# Spies: prove the unified kernel actually compiled + launched.
# --------------------------------------------------------------------------- #
class UnifiedSpy:
    """Counts (a) real unified cubin launches via sparkinfer._lib.compiler.launch, and
    (b) dispatch-entrypoint calls into run_unified_decode / run_unified_prefill.

    Both are installed as wrappers around the genuine implementations (pass-through
    -- the kernel still runs), so timing is unaffected and a fallback to legacy is
    detectable (the counters stay flat)."""

    def __init__(self) -> None:
        self.kernel_launches = 0          # SM120 cubin launches (compiler.launch)
        self.decode_dispatch = 0          # run_unified_decode entrypoint hits
        self.prefill_dispatch = 0         # run_unified_prefill entrypoint hits
        self._saved: list[tuple] = []

    def reset(self) -> None:
        self.kernel_launches = 0
        self.decode_dispatch = 0
        self.prefill_dispatch = 0

    def install(self) -> None:
        import sparkinfer._lib.compiler as compiler_mod
        import sparkinfer.attention._shared.mla.kernel as launch_mod
        import sparkinfer.attention._shared.mla.prefill as prefill_mod

        real_launch = compiler_mod.launch

        def spy_launch(func, *, compile_spec, compile_args, runtime_args, compile_kwargs=None):
            kid = str(getattr(compile_spec, "kernel_id", ""))
            if "sm120" in kid:
                self.kernel_launches += 1
            return real_launch(
                func,
                compile_spec=compile_spec,
                compile_args=compile_args,
                runtime_args=runtime_args,
                compile_kwargs=compile_kwargs,
            )

        # launch.py captured `launch as sparkinfer_launch` at import; patch that binding
        # too so the decode launcher's own reference is intercepted.
        self._saved.append((compiler_mod, "launch", compiler_mod.launch))
        compiler_mod.launch = spy_launch
        if hasattr(launch_mod, "sparkinfer_launch"):
            self._saved.append((launch_mod, "sparkinfer_launch", launch_mod.sparkinfer_launch))
            launch_mod.sparkinfer_launch = spy_launch
        if hasattr(prefill_mod, "sparkinfer_launch"):
            self._saved.append((prefill_mod, "sparkinfer_launch", prefill_mod.sparkinfer_launch))
            prefill_mod.sparkinfer_launch = spy_launch

        # Dispatch-entrypoint wrappers. The APIs import the promoted root
        # ``kernel.py`` entrypoints inside the call. Patch that binding.
        real_decode = launch_mod.run_unified_decode

        def spy_decode(*args, **kwargs):
            self.decode_dispatch += 1
            return real_decode(*args, **kwargs)

        self._saved.append((launch_mod, "run_unified_decode", launch_mod.run_unified_decode))
        launch_mod.run_unified_decode = spy_decode

        real_prefill = prefill_mod.run_unified_prefill

        def spy_prefill(*args, **kwargs):
            self.prefill_dispatch += 1
            return real_prefill(*args, **kwargs)

        self._saved.append((prefill_mod, "run_unified_prefill", prefill_mod.run_unified_prefill))
        prefill_mod.run_unified_prefill = spy_prefill

    def uninstall(self) -> None:
        for obj, name, value in reversed(self._saved):
            setattr(obj, name, value)
        self._saved.clear()


@contextlib.contextmanager
def unified_enabled(enabled: bool):
    prev = os.environ.get(_UNIFIED_ENV)
    if enabled:
        os.environ[_UNIFIED_ENV] = "1"
    else:
        os.environ[_UNIFIED_ENV] = "0"
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop(_UNIFIED_ENV, None)
        else:
            os.environ[_UNIFIED_ENV] = prev


# --------------------------------------------------------------------------- #
# Timing helper: capture a CUDA graph, count unified launches during
# warmup+capture, then time replays.
# --------------------------------------------------------------------------- #
def _time_graph(run, *, spy: UnifiedSpy, warmup: int, replays: int, l2_flush):
    import sparkinfer.attention._shared.mla.kernel as _launch_mod

    spy.reset()
    _launch_mod.LAST_DECODE_PLAN.clear()
    graph = capture_cuda_graph(run, warmup=warmup)
    # snapshot the spy AFTER warmup+capture: any unified cubin / dispatch hit is
    # now recorded.  (Replays do not re-enter Python, so this is the proof.)
    kernel_launches = spy.kernel_launches
    decode_dispatch = spy.decode_dispatch
    prefill_dispatch = spy.prefill_dispatch
    # The launcher records the chosen wave-balanced split plan as a side-channel
    # (LAST_DECODE_PLAN). For the unified run this is the num_splits actually used
    # in the captured graph; for the legacy run the dict stays empty.
    num_splits_used = int(_launch_mod.LAST_DECODE_PLAN.get("num_splits", 0) or 0)
    try:
        stats = bench_cuda_graph(graph, replays=replays, l2_flush=l2_flush)
    finally:
        torch.cuda.synchronize()
        del graph
        gc.collect()
        torch.cuda.empty_cache()
    replay_us = stats["replay_us"]
    return {
        "median_us": statistics.median(replay_us),
        "min_us": min(replay_us),
        "kernel_launches": kernel_launches,
        "decode_dispatch": decode_dispatch,
        "prefill_dispatch": prefill_dispatch,
        "num_splits_used": num_splits_used,
    }


# --------------------------------------------------------------------------- #
# DSV4 compressed decode (q_head_dim = 512).
# --------------------------------------------------------------------------- #
def _make_dsv4_inputs(*, rows, num_heads, topk, device, seed):
    gen = torch.Generator(device=device).manual_seed(seed)
    # A realistic compressed page pool: enough tokens to back the topk selection.
    page_size = COMPRESSED_MLA_DSV4_PAGE_SIZE  # 256 (the real DSV4 main-cache page)
    n_tokens = max(topk * 2, page_size * 4)
    n_tokens = ((n_tokens + page_size - 1) // page_size) * page_size
    num_pages = n_tokens // page_size
    k_nope = (torch.randn((n_tokens, COMPRESSED_MLA_NOPE_DIM), generator=gen,
                          dtype=torch.float32, device=device) / 10).clamp(-1, 1)
    k_rope = (torch.randn((n_tokens, COMPRESSED_MLA_ROPE_DIM), generator=gen,
                          dtype=torch.float32, device=device) / 10).clamp(-1, 1)
    cache = pack_compressed_mla_kv_cache_reference(
        k_nope, k_rope.to(torch.bfloat16), page_size=page_size, num_pages=num_pages
    )
    q = (torch.randn((rows, num_heads, _DSV4_HEAD_DIM), generator=gen,
                     dtype=torch.float32, device=device) / 10).clamp(-1, 1).to(torch.bfloat16)
    idx = torch.randint(0, n_tokens, (rows, topk), generator=gen,
                        dtype=torch.int32, device=device)
    lengths = torch.full((rows,), topk, dtype=torch.int32, device=device)
    page_nbytes = compressed_mla_page_nbytes(page_size)
    # KV bytes touched per query row (the bandwidth-bound term for decode): topk
    # candidate tokens * per-token compressed record bytes.
    per_token_bytes = page_nbytes / page_size
    kv_bytes = int(rows * topk * per_token_bytes)
    return q, cache, idx, lengths, page_size, kv_bytes


def _make_dsv4_workspace(*, rows, num_heads, topk, device, max_chunks):
    return SPARKINFERAttentionWorkspace.for_fixed_capacity(
        mode="decode",
        device=device,
        dtype=torch.bfloat16,
        kv_dtype=torch.uint8,
        num_q_heads=num_heads,
        head_dim=_DSV4_HEAD_DIM,
        v_head_dim=_DSV4_HEAD_DIM,
        topk=topk,
        max_total_q=rows,
        max_batch=rows,
        page_size=COMPRESSED_MLA_DSV4_PAGE_SIZE,
        use_cuda_graph=True,
        max_chunks_per_row=max_chunks,
        reserve_compressed_mla_staging=True,
    )


def bench_dsv4_decode(*, rows, num_heads, topk, device, spy, warmup, replays, l2_flush, seed):
    sm_scale = 1.0 / math.sqrt(_DSV4_HEAD_DIM)
    q, cache, idx, lengths, page_size, kv_bytes = _make_dsv4_inputs(
        rows=rows, num_heads=num_heads, topk=topk, device=device, seed=seed,
    )
    max_chunks = max(1, (topk + 63) // 64)

    def make_run(enabled):
        ws = _make_dsv4_workspace(
            rows=rows, num_heads=num_heads, topk=topk, device=device, max_chunks=max_chunks,
        )

        def run():
            return compressed_mla_decode_forward(
                q_all=q,
                swa_k_cache=cache,
                swa_indices=idx,
                swa_topk_lengths=lengths,
                workspace=ws,
                sm_scale=sm_scale,
                swa_page_size=page_size,
            )
        return run

    clear_mla_caches()
    with unified_enabled(False):
        legacy = _time_graph(make_run(False), spy=spy, warmup=warmup, replays=replays, l2_flush=l2_flush)
    clear_mla_caches()
    with unified_enabled(True):
        unified = _time_graph(make_run(True), spy=spy, warmup=warmup, replays=replays, l2_flush=l2_flush)

    return _assemble(
        config=f"DSV4-decode heads={num_heads} q={_DSV4_HEAD_DIM} topk={topk} rows={rows}",
        legacy=legacy, unified=unified, kv_bytes=kv_bytes, kind="decode",
    )


# --------------------------------------------------------------------------- #
# GLM sparse decode (q_head_dim = 576, 656B/token KV).
# --------------------------------------------------------------------------- #
def _make_glm_inputs(*, rows, num_heads, topk, device, seed):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    n_tokens = max(topk * 2, 1024)
    k_nope = (torch.randn((n_tokens, 1, _MLA_NOPE_DIM), generator=gen, dtype=torch.float32)
              .div_(4.0).to(torch.bfloat16).to(device))
    k_rope = (torch.randn((n_tokens, 1, _MLA_ROPE_DIM), generator=gen, dtype=torch.float32)
              .div_(4.0).to(torch.bfloat16).to(device))
    kv_cache = pack_mla_kv_cache_reference(k_nope, k_rope)   # (n_tokens, 1, 656) uint8
    assert kv_cache.shape[-1] == _GLM_KV_BYTES_PER_TOKEN, kv_cache.shape
    q = (torch.randn((rows, num_heads, _GLM_Q_HEAD_DIM), generator=gen, dtype=torch.float32)
         .div_(4.0).to(torch.bfloat16).to(device))
    page_table = torch.randint(0, n_tokens, (rows, topk), generator=gen,
                               dtype=torch.int32).to(device)
    cache_seqlens = torch.full((rows,), topk, dtype=torch.int32, device=device)
    kv_bytes = int(rows * topk * _GLM_KV_BYTES_PER_TOKEN)
    return q, kv_cache, page_table, cache_seqlens, kv_bytes


def _make_glm_workspace(*, rows, num_heads, topk, device):
    # page_size=1 so the unified GLM gather treats each selected index as a single
    # 656B token record (stride_kv_block = page_size * 656). The legacy sparse
    # decode reads the same (n_tokens,1,656) cache with token-level page_table_1.
    return SPARKINFERAttentionWorkspace.for_fixed_capacity(
        mode="decode",
        device=device,
        dtype=torch.bfloat16,
        kv_dtype=torch.uint8,
        num_q_heads=num_heads,
        head_dim=_GLM_Q_HEAD_DIM,
        v_head_dim=_GLM_V_HEAD_DIM,
        topk=topk,
        max_total_q=rows,
        max_batch=rows,
        max_kv_rows=rows * topk,
        page_size=1,
        use_cuda_graph=True,
        max_chunks_per_row=max(1, (topk + 63) // 64),
    )


def bench_glm_decode(*, rows, num_heads, topk, device, spy, warmup, replays, l2_flush, seed):
    sm_scale = 1.0 / math.sqrt(_GLM_Q_HEAD_DIM)
    q, kv_cache, page_table, cache_seqlens, kv_bytes = _make_glm_inputs(
        rows=rows, num_heads=num_heads, topk=topk, device=device, seed=seed,
    )

    def make_run():
        ws = _make_glm_workspace(rows=rows, num_heads=num_heads, topk=topk, device=device)

        def run():
            return sparse_mla_decode_forward(
                q_all=q,
                kv_cache=kv_cache,
                page_table_1=page_table,
                cache_seqlens_int32=cache_seqlens,
                nsa_cache_seqlens_int32=cache_seqlens,
                workspace=ws,
                sm_scale=sm_scale,
                v_head_dim=_GLM_V_HEAD_DIM,
            )
        return run

    clear_mla_caches()
    with unified_enabled(False):
        legacy = _time_graph(make_run(), spy=spy, warmup=warmup, replays=replays, l2_flush=l2_flush)
    clear_mla_caches()
    with unified_enabled(True):
        unified = _time_graph(make_run(), spy=spy, warmup=warmup, replays=replays, l2_flush=l2_flush)

    return _assemble(
        config=f"GLM-decode heads={num_heads} q={_GLM_Q_HEAD_DIM} topk={topk} rows={rows}",
        legacy=legacy, unified=unified, kv_bytes=kv_bytes, kind="decode",
    )


# --------------------------------------------------------------------------- #
# DSV4 prefill (unified run_unified_prefill).  Legacy comparable: the legacy
# compressed split-decode over the same T prefill rows (no dedicated legacy
# compressed prefill kernel exists).
# --------------------------------------------------------------------------- #
def _import_prefill_ref():
    sm120port = str(pathlib.Path(__file__).resolve().parents[1] / ".sm120port")
    if sm120port not in sys.path:
        sys.path.insert(0, sm120port)
    import dsv4_ref  # noqa: F401
    from tests._reference import prefill_ref
    return prefill_ref, dsv4_ref


def _repack_prefill_to_compressed(packed_dsv4, page_size, num_blocks, dsv4_ref):
    import sparkinfer.attention._shared.mla.compressed_reference as cr
    bpt = dsv4_ref.DSV4_KV_GMEM_STRIDE  # 584
    page_nbytes = cr.compressed_mla_page_nbytes(page_size)
    flat = packed_dsv4.reshape(num_blocks, page_size * bpt)
    out = torch.zeros(num_blocks, page_nbytes, dtype=torch.uint8, device=packed_dsv4.device)
    out[:, : page_size * bpt] = flat
    return out


def bench_dsv4_prefill(*, num_tokens, num_heads, topk, device, spy, warmup, replays, l2_flush, seed):
    prefill_ref, dsv4_ref = _import_prefill_ref()
    from sparkinfer.attention._shared.mla.prefill import run_unified_prefill as _direct_prefill

    page_size = _DSV4_PREFILL_PAGE
    num_blocks = max(2, (topk + page_size - 1) // page_size + 1)
    case = prefill_ref.make_dsv4_prefill_case(
        num_tokens=num_tokens, num_heads=num_heads, topk=topk,
        num_blocks=num_blocks, page_block_size=page_size,
        with_sink=False, invalidate_half=True, device=device, seed=seed,
    )
    q = case["q"].contiguous()
    idx = case["topk_indices"].contiguous()
    lengths = case["topk_lengths"].contiguous()
    sm_scale = case["sm_scale"]
    swa_cache = _repack_prefill_to_compressed(case["kv_cache"], page_size, num_blocks, dsv4_ref)

    # KV bytes touched: T tokens * topk candidates * per-token compressed record.
    per_token_bytes = compressed_mla_page_nbytes(page_size) / page_size
    kv_bytes = int(num_tokens * topk * per_token_bytes)

    # ---- UNIFIED prefill (the kernel under test). The dispatch wrapper proves
    #      run_unified_prefill ran; the launch counter proves the cubin launched.
    #      Call the SPIED package entrypoint (prefill_mod.run_unified_prefill) so
    #      prefill_dispatch is counted -- _direct_prefill above is only imported to
    #      assert the symbol exists.
    del _direct_prefill
    import sparkinfer.attention._shared.mla.prefill as prefill_mod

    def run_unified():
        return prefill_mod.run_unified_prefill(
            q=q, kv_cache=swa_cache, topk_indices=idx, sm_scale=sm_scale,
            page_block_size=page_size, topk_length=lengths,
        )

    with unified_enabled(True):
        unified = _time_graph(run_unified, spy=spy, warmup=warmup, replays=replays, l2_flush=l2_flush)

    # ---- LEGACY comparable: legacy compressed split-decode over the same T rows.
    #      Same compressed cache, same topk indices, flag OFF.  This is the
    #      closest existing legacy compressed kernel (no dedicated legacy prefill).
    clear_mla_caches()
    legacy = None
    legacy_note = "legacy=compressed split-decode over T prefill rows (no dedicated legacy compressed prefill kernel)"
    try:
        ws = _make_dsv4_workspace(
            rows=num_tokens, num_heads=num_heads, topk=topk, device=device,
            max_chunks=max(1, (topk + 63) // 64),
        )
        legacy_q = q.contiguous()
        legacy_lengths = lengths.to(torch.int32).contiguous()

        def run_legacy():
            return compressed_mla_decode_forward(
                q_all=legacy_q,
                swa_k_cache=swa_cache,
                swa_indices=idx,
                swa_topk_lengths=legacy_lengths,
                workspace=ws,
                sm_scale=sm_scale,
                swa_page_size=page_size,
            )
        with unified_enabled(False):
            legacy = _time_graph(run_legacy, spy=spy, warmup=warmup, replays=replays, l2_flush=l2_flush)
    except Exception as exc:  # pragma: no cover - baseline best-effort
        legacy_note += f" | legacy baseline unavailable: {type(exc).__name__}: {exc}"

    return _assemble_prefill(
        config=f"DSV4-prefill heads={num_heads} q={_DSV4_HEAD_DIM} topk={topk} T={num_tokens}",
        legacy=legacy, unified=unified, kv_bytes=kv_bytes, note=legacy_note,
    )


# --------------------------------------------------------------------------- #
# Result assembly.
# --------------------------------------------------------------------------- #
def _bw_note(kv_bytes, median_us):
    if median_us <= 0:
        return ""
    gbps = kv_bytes / (median_us * 1e-6) / 1e9
    return f"KV {kv_bytes/1e6:.2f} MB -> {gbps:.0f} GB/s @ {median_us:.1f}us"


def _assemble(*, config, legacy, unified, kv_bytes, kind):
    unified_ran = unified["kernel_launches"] > 0 and unified["decode_dispatch"] > 0
    legacy_us = legacy["median_us"]
    unified_us = unified["median_us"]
    speedup = legacy_us / unified_us if unified_us > 0 else 0.0
    return {
        "config": config,
        "legacy_us": round(legacy_us, 3),
        "unified_us": round(unified_us, 3),
        "legacy_min_us": round(legacy["min_us"], 3),
        "unified_min_us": round(unified["min_us"], 3),
        "speedup": round(speedup, 3),
        "num_splits_used": unified.get("num_splits_used", 0),
        "unified_actually_ran": bool(unified_ran),
        "unified_kernel_launches": unified["kernel_launches"],
        "unified_decode_dispatch": unified["decode_dispatch"],
        "legacy_kernel_launches": legacy["kernel_launches"],
        "kv_bandwidth_note": "; ".join(
            x for x in (
                "legacy " + _bw_note(kv_bytes, legacy_us),
                "unified " + _bw_note(kv_bytes, unified_us),
            ) if x.strip() != "legacy" and x.strip() != "unified"
        ),
    }


def _assemble_prefill(*, config, legacy, unified, kv_bytes, note):
    unified_ran = unified["kernel_launches"] > 0 and unified["prefill_dispatch"] > 0
    unified_us = unified["median_us"]
    if legacy is not None:
        legacy_us = legacy["median_us"]
        speedup = legacy_us / unified_us if unified_us > 0 else 0.0
        legacy_launches = legacy["kernel_launches"]
    else:
        legacy_us = 0.0
        speedup = 0.0
        legacy_launches = 0
    bw = "unified " + _bw_note(kv_bytes, unified_us)
    if legacy is not None:
        bw = "legacy " + _bw_note(kv_bytes, legacy_us) + "; " + bw
    return {
        "config": config,
        "legacy_us": round(legacy_us, 3),
        "unified_us": round(unified_us, 3),
        "unified_min_us": round(unified["min_us"], 3),
        "speedup": round(speedup, 3),
        "unified_actually_ran": bool(unified_ran),
        "unified_kernel_launches": unified["kernel_launches"],
        "unified_prefill_dispatch": unified["prefill_dispatch"],
        "legacy_kernel_launches": legacy_launches,
        "kv_bandwidth_note": bw,
        "note": note,
    }


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--replays", type=int, default=80)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--dsv4-heads", type=int, default=128)
    parser.add_argument("--glm-heads", type=int, default=128)
    parser.add_argument("--dsv4-topk", type=str, default="512,1024,2048")
    parser.add_argument("--glm-topk", type=str, default="512,1024,2048")
    parser.add_argument("--dsv4-rows", type=int, default=1)
    parser.add_argument("--glm-rows", type=int, default=1)
    parser.add_argument("--prefill-tokens", type=str, default="2048")
    parser.add_argument("--prefill-topk", type=str, default="512,1024,2048")
    parser.add_argument("--no-flush-l2", action="store_false", dest="flush_l2", default=True)
    parser.add_argument("--skip-prefill", action="store_true")
    args = parser.parse_args(argv)

    if args.warmup < 10:
        raise SystemExit("--warmup must be >= 10")
    if args.replays < 50:
        raise SystemExit("--replays must be >= 50")

    device = require_sm120()
    l2_flush = make_l2_flush_fn(args.flush_l2, resolve_l2_flush_bytes(0))

    spy = UnifiedSpy()
    spy.install()
    results = []
    try:
        for topk in [int(x) for x in args.dsv4_topk.split(",") if x]:
            print(f"[run] DSV4 decode topk={topk} heads={args.dsv4_heads} rows={args.dsv4_rows}", flush=True)
            results.append(bench_dsv4_decode(
                rows=args.dsv4_rows, num_heads=args.dsv4_heads, topk=topk, device=device,
                spy=spy, warmup=args.warmup, replays=args.replays, l2_flush=l2_flush,
                seed=args.seed + topk,
            ))
        for topk in [int(x) for x in args.glm_topk.split(",") if x]:
            print(f"[run] GLM decode topk={topk} heads={args.glm_heads} rows={args.glm_rows}", flush=True)
            try:
                results.append(bench_glm_decode(
                    rows=args.glm_rows, num_heads=args.glm_heads, topk=topk, device=device,
                    spy=spy, warmup=args.warmup, replays=args.replays, l2_flush=l2_flush,
                    seed=args.seed + 7 + topk,
                ))
            except Exception as exc:
                print(f"[warn] GLM decode topk={topk} failed: {type(exc).__name__}: {exc}", flush=True)
                results.append({"config": f"GLM-decode heads={args.glm_heads} topk={topk}",
                                "error": f"{type(exc).__name__}: {exc}",
                                "legacy_us": 0.0, "unified_us": 0.0, "speedup": 0.0,
                                "unified_actually_ran": False})
        if not args.skip_prefill:
            for tokens in [int(x) for x in args.prefill_tokens.split(",") if x]:
                for topk in [int(x) for x in args.prefill_topk.split(",") if x]:
                    print(f"[run] DSV4 prefill T={tokens} topk={topk} heads={args.dsv4_heads}", flush=True)
                    try:
                        results.append(bench_dsv4_prefill(
                            num_tokens=tokens, num_heads=args.dsv4_heads, topk=topk, device=device,
                            spy=spy, warmup=args.warmup, replays=args.replays, l2_flush=l2_flush,
                            seed=args.seed + 99 + topk,
                        ))
                    except Exception as exc:
                        print(f"[warn] prefill T={tokens} topk={topk} failed: {type(exc).__name__}: {exc}", flush=True)
                        results.append({"config": f"DSV4-prefill heads={args.dsv4_heads} T={tokens} topk={topk}",
                                        "error": f"{type(exc).__name__}: {exc}",
                                        "legacy_us": 0.0, "unified_us": 0.0, "speedup": 0.0,
                                        "unified_actually_ran": False})
    finally:
        spy.uninstall()

    print("\n==================== UNIFIED vs LEGACY (graph-captured) ====================")
    hdr = f"{'config':<52} {'legacy_us':>10} {'unified_us':>11} {'speedup':>8} {'splits':>7} {'unified_ran':>12}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        if "error" in r:
            print(f"{r['config']:<52} {'ERROR':>10} {r['error']}")
            continue
        print(f"{r['config']:<52} {r['legacy_us']:>10.2f} {r['unified_us']:>11.2f} "
              f"{r['speedup']:>8.3f} {r.get('num_splits_used', 0):>7} "
              f"{str(r['unified_actually_ran']):>12}")
    print("\n--- KV bandwidth ---")
    for r in results:
        if "error" in r:
            continue
        print(f"{r['config']:<52} {r.get('kv_bandwidth_note','')}")
        if "note" in r:
            print(f"{'':<52} {r['note']}")

    import json
    print("\nJSON:")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
