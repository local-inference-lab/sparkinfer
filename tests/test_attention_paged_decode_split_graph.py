"""Regression tests for force-split dense decode under CUDA graphs.

Covers the gqa_group_size > 8 case (e.g. MiniMax-M3 dense layers: 64 q heads /
4 KV heads = gqa 16, head_dim 128) where the single-qtile / regularized decode
graph epilogues previously only stored MMA row_slot 0 (packed rows 0-7),
dropping q heads 8..group_size-1 of every KV group.

Note the CUDA-graph metadata contract exercised here: with
copy_runtime_metadata (the default), the copy from the caller's metadata
tensors into the scratch's static buffers is captured inside the graph, so the
metadata tensors passed at capture must keep stable addresses across replays —
callers update their contents in place (this is what a serving engine does with
persistent buffers).
"""
from __future__ import annotations

import torch

from sparkinfer.attention.paged.reference import paged_attention_reference
from sparkinfer.integration.attention import (
    SPARKINFERPagedAttentionScratchCaps,
    clear_attention_caches,
    create_paged_plan,
    paged_attention_forward,
    plan_paged_attention_scratch,
)

from .helpers import require_sm12x
from .paged_attention_helpers import make_paged_inputs


class _ForcedSplitDecodeGraphHarness:
    """Decode-graph harness with split-KV forced on (dense split decode)."""

    def __init__(self, q, k_cache, v_cache):
        self.q = q
        self.k_cache = k_cache
        self.v_cache = v_cache
        self._scratch_plan = None
        self._scratch = None
        self._output = None
        self._page_table = None
        self._cache_seqlens = None
        self._cu_seqlens_q = None
        self._last_scratch = None

    def prepare(self, page_table, cache_seqlens, cu_seqlens_q):
        plan = create_paged_plan(
            self.q,
            self.k_cache,
            self.v_cache,
            page_table,
            cache_seqlens,
            cu_seqlens_q,
            mode="decode",
            enable_cuda_graph=True,
            force_split_kv=True,
        )
        if self._scratch_plan is None:
            batch = page_table.shape[0]
            self._scratch_plan = plan_paged_attention_scratch(
                SPARKINFERPagedAttentionScratchCaps(
                    device=self.q.device,
                    mode="decode",
                    dtype=self.q.dtype,
                    kv_dtype=self.k_cache.dtype,
                    num_q_heads=self.q.shape[1],
                    num_kv_heads=self.k_cache.shape[2],
                    head_dim_qk=self.q.shape[2],
                    head_dim_vo=self.v_cache.shape[3],
                    page_size=self.k_cache.shape[1],
                    max_total_q=plan.total_q,
                    max_batch=batch,
                    max_page_table_width=page_table.shape[1],
                    # split decode graphs pad work items to the CTA policy
                    # (~2 CTAs/SM), so size generously rather than from the
                    # bootstrap plan
                    max_work_items=max(plan.new_batch_size, 2048),
                    max_partial_rows=max(plan.total_num_partial_rows, 2048),
                    num_cache_pages=self.k_cache.shape[0],
                    use_cuda_graph=True,
                )
            )
            self._scratch = tuple(
                torch.empty(shape, dtype=dtype, device=self.q.device)
                for shape, dtype in self._scratch_plan.shapes_and_dtypes()
            )
            self._scratch_plan.prepare_decode_graph_replay_state(
                batch=batch,
                max_page_table_width=page_table.shape[1],
                force_split_kv=True,
            )
        self._page_table = page_table
        self._cache_seqlens = cache_seqlens
        self._cu_seqlens_q = cu_seqlens_q
        if self._output is not None:
            self._bind()

    def _bind(self):
        binding = self._scratch_plan.bind(
            scratch=self._scratch,
            q=self.q,
            k_cache=self.k_cache,
            v_cache=self.v_cache,
            output=self._output,
            page_table=self._page_table,
            cache_seqlens=self._cache_seqlens,
            cu_seqlens_q=self._cu_seqlens_q,
            k_descale=None,
            v_descale=None,
        )
        self._last_scratch = binding.scratch
        return binding

    def run(self, output):
        self._output = output
        return paged_attention_forward(binding=self._bind())


def _run_forced_split_decode_graph(
    *,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    cache_seqlens_capture: list[int],
    cache_seqlens_replay: list[int] | None = None,
    seed: int = 73,
) -> None:
    clear_attention_caches()
    batch = len(cache_seqlens_capture)
    max_len = max(cache_seqlens_capture + (cache_seqlens_replay or []))
    page_size = 64
    width = max_len // page_size + 2
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = make_paged_inputs(
        q_seqlens=[1] * batch,
        cache_seqlens=cache_seqlens_capture,
        page_size=page_size,
        seed=seed,
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        page_table_width=width,
        num_pages=4096,
    )
    harness = _ForcedSplitDecodeGraphHarness(q, k_cache, v_cache)
    harness.prepare(page_table, cache_seqlens, cu_seqlens_q)
    output = torch.empty_like(q)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        harness.run(output)

    ref_out, _ = paged_attention_reference(
        q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, causal=True
    )
    graph.replay()
    torch.cuda.synchronize()
    assert (output - ref_out).abs().max().item() <= 0.02

    if cache_seqlens_replay is None:
        return

    q2, k2, v2, pt2, csl2, cu2 = make_paged_inputs(
        q_seqlens=[1] * batch,
        cache_seqlens=cache_seqlens_replay,
        page_size=page_size,
        seed=seed + 6,
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        page_table_width=page_table.shape[1],
        num_pages=k_cache.shape[0],
    )
    # keep metadata tensor addresses stable; update contents in place
    q.copy_(q2)
    k_cache.copy_(k2)
    v_cache.copy_(v2)
    page_table.copy_(pt2)
    cache_seqlens.copy_(csl2)
    cu_seqlens_q.copy_(cu2)
    harness.prepare(page_table, cache_seqlens, cu_seqlens_q)

    ref_out2, _ = paged_attention_reference(
        q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, causal=True
    )
    graph.replay()
    torch.cuda.synchronize()
    assert (output - ref_out2).abs().max().item() <= 0.02


@torch.inference_mode()
def test_forced_split_decode_graph_gqa16_head_dim128() -> None:
    """MiniMax-M3 dense-layer shape: 64 q heads / 4 KV heads / head_dim 128."""
    require_sm12x()
    _run_forced_split_decode_graph(
        q_heads=64,
        kv_heads=4,
        head_dim=128,
        cache_seqlens_capture=[4096, 4096, 4096, 4096],
        cache_seqlens_replay=[512, 1536, 3072, 4096],
    )


@torch.inference_mode()
def test_forced_split_decode_graph_gqa12_head_dim128() -> None:
    """gqa 12: second MMA row slot partially occupied."""
    require_sm12x()
    _run_forced_split_decode_graph(
        q_heads=48,
        kv_heads=4,
        head_dim=128,
        cache_seqlens_capture=[4096, 4096, 4096, 4096],
    )


@torch.inference_mode()
def test_forced_split_decode_graph_gqa8_head_dim256() -> None:
    """gqa 8 / head_dim 256 control (single row slot fast path)."""
    require_sm12x()
    _run_forced_split_decode_graph(
        q_heads=8,
        kv_heads=1,
        head_dim=256,
        cache_seqlens_capture=[4096, 4096, 4096, 4096],
        cache_seqlens_replay=[512, 1536, 3072, 4096],
    )


@torch.inference_mode()
def test_forced_split_decode_graph_gqa16_head_dim128_batch2() -> None:
    """batch != 4 exercises the bf16/128 regular-decode merge bdy=3 config."""
    require_sm12x()
    _run_forced_split_decode_graph(
        q_heads=64,
        kv_heads=4,
        head_dim=128,
        cache_seqlens_capture=[4096, 2048],
        cache_seqlens_replay=[1024, 4096],
    )
