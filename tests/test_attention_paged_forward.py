from __future__ import annotations

import math

import pytest
import torch

from benchmarks.benchmark_paged_attention import (
    _capture_backend_graph,
    _capture_flashinfer_fa2_graph,
    _make_uniform_paged_inputs,
    _quantize_paged_kv_cache_global_e4m3,
)
from b12x.attention.paged.reference import paged_attention_reference
from b12x.integration.attention import (
    B12XPagedAttentionScratchCaps,
    clear_attention_caches,
    create_paged_plan,
    paged_attention_forward,
    plan_paged_attention_scratch,
)

from .helpers import require_sm120
from .paged_attention_helpers import quantize_paged_kv_cache_e4m3
from .test_attention_paged_planner import _make_inputs


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.to(torch.float32).reshape(-1)
    b_f = b.to(torch.float32).reshape(-1)
    return torch.nn.functional.cosine_similarity(a_f, b_f, dim=0).item()


class _PagedScratchHarness:
    def __init__(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        *,
        mode: str,
    ) -> None:
        self.q = q
        self.k_cache = k_cache
        self.v_cache = v_cache
        self.mode = mode
        self.plan = None
        self._scratch_plan = None
        self._scratch = None
        self._prepare_kwargs = {}
        self._page_table = None
        self._cache_seqlens = None
        self._cu_seqlens_q = None

    def prepare(
        self,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        **kwargs,
    ) -> None:
        self.plan = create_paged_plan(
            self.q,
            self.k_cache,
            self.v_cache,
            page_table,
            cache_seqlens,
            cu_seqlens_q,
            mode=self.mode,
            **kwargs,
        )
        self._scratch_plan = plan_paged_attention_scratch(
            B12XPagedAttentionScratchCaps(
                device=self.q.device,
                mode=self.mode,
                dtype=self.q.dtype,
                kv_dtype=self.k_cache.dtype,
                num_q_heads=self.q.shape[1],
                num_kv_heads=self.k_cache.shape[2],
                head_dim_qk=self.q.shape[2],
                head_dim_vo=self.v_cache.shape[3],
                page_size=self.k_cache.shape[1],
                max_total_q=self.plan.total_q,
                max_batch=page_table.shape[0],
                max_page_table_width=page_table.shape[1],
                max_work_items=max(self.plan.new_batch_size, 1),
                max_partial_rows=self.plan.total_num_partial_rows,
                num_cache_pages=self.k_cache.shape[0],
            )
        )
        self._scratch = tuple(
            torch.empty(shape, dtype=dtype, device=self.q.device)
            for shape, dtype in self._scratch_plan.shapes_and_dtypes()
        )
        self._prepare_kwargs = dict(kwargs)
        self._page_table = page_table
        self._cache_seqlens = cache_seqlens
        self._cu_seqlens_q = cu_seqlens_q

    def run(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        *,
        output: torch.Tensor,
        k_descale: torch.Tensor | None = None,
        v_descale: torch.Tensor | None = None,
        attention_sink_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self._scratch_plan is not None
        assert self._scratch is not None
        binding = self._scratch_plan.bind(
            scratch=self._scratch,
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            output=output,
            page_table=self._page_table,
            cache_seqlens=self._cache_seqlens,
            cu_seqlens_q=self._cu_seqlens_q,
            k_descale=k_descale,
            v_descale=v_descale,
            attention_sink_bias=attention_sink_bias,
            **self._prepare_kwargs,
        )
        self.plan = binding.scratch.plan
        return paged_attention_forward(binding=binding)


def _make_workspace(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    mode: str,
) -> _PagedScratchHarness:
    return _PagedScratchHarness(q, k_cache, v_cache, mode=mode)


def _run_decode_graph_check(
    *,
    batch: int = 8,
    cache_seqlen: int,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    (
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        capture_page_table,
        capture_cache_seqlens,
        cu_seqlens_q,
    ) = _make_uniform_paged_inputs(
        batch=batch,
        q_seqlen=1,
        cache_seqlen=cache_seqlen,
        capture_cache_seqlen=None,
        page_size=64,
        q_heads=8,
        kv_heads=1,
        head_dim=256,
        dtype=torch.bfloat16,
        seed=1,
    )
    k_fp8, v_fp8, k_descale, v_descale, k_scale, v_scale = _quantize_paged_kv_cache_global_e4m3(
        k_cache,
        v_cache,
        batch=batch,
        kv_heads=1,
    )
    backend = _capture_backend_graph(
        q=q,
        k_cache=k_fp8,
        v_cache=v_fp8,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        capture_page_table=capture_page_table,
        capture_cache_seqlens=capture_cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        fixed_split_pages=None,
        k_descale=k_descale,
        v_descale=v_descale,
        warmup=1,
        graph_ctas_per_sm=None,
    )
    _fa2_graph, fa2_out = _capture_flashinfer_fa2_graph(
        q=q,
        k_cache=k_fp8,
        v_cache=v_fp8,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        capture_page_table=capture_page_table,
        capture_cache_seqlens=capture_cache_seqlens,
        q_seqlen=1,
        page_size=64,
        q_heads=8,
        kv_heads=1,
        head_dim=256,
        q_dtype=torch.bfloat16,
        kv_dtype=torch.float8_e4m3fn,
        k_scale=k_scale,
        v_scale=v_scale,
        workspace_bytes=512 * 1024 * 1024,
        warmup=1,
    )
    return backend.output, fa2_out, backend.plan_desc


def _run_decode_reference_check(
    *,
    batch: int = 8,
    cache_seqlen: int,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    (
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        capture_page_table,
        capture_cache_seqlens,
        cu_seqlens_q,
    ) = _make_uniform_paged_inputs(
        batch=batch,
        q_seqlen=1,
        cache_seqlen=cache_seqlen,
        capture_cache_seqlen=None,
        page_size=64,
        q_heads=8,
        kv_heads=1,
        head_dim=256,
        dtype=torch.bfloat16,
        seed=1,
    )
    k_fp8, v_fp8, k_descale, v_descale, _k_scale, _v_scale = _quantize_paged_kv_cache_global_e4m3(
        k_cache,
        v_cache,
        batch=batch,
        kv_heads=1,
    )
    backend = _capture_backend_graph(
        q=q,
        k_cache=k_fp8,
        v_cache=v_fp8,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        capture_page_table=capture_page_table,
        capture_cache_seqlens=capture_cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        fixed_split_pages=None,
        k_descale=k_descale,
        v_descale=v_descale,
        warmup=1,
        graph_ctas_per_sm=None,
    )
    backend.graph.replay()
    torch.cuda.synchronize()
    ref_out, _ref_lse = paged_attention_reference(
        q,
        k_fp8,
        v_fp8,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        k_descale=k_descale,
        v_descale=v_descale,
        causal=True,
    )
    return backend.output, ref_out, backend.plan_desc


@torch.inference_mode()
def test_paged_forward_matches_reference_decode_short_context() -> None:
    require_sm120()
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
        q_seqlens=[1, 1, 1],
        cache_seqlens=[64, 128, 192],
        dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
    )
    workspace = _make_workspace(q, k_cache, v_cache, mode="decode")
    workspace.prepare(page_table, cache_seqlens, cu_seqlens_q)
    output, lse_base2 = workspace.run(
        q,
        k_cache,
        v_cache,
        output=torch.empty_like(q),
    )
    torch.cuda.synchronize()

    ref_out, ref_lse = paged_attention_reference(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        causal=True,
    )
    lse_natural = lse_base2 * math.log(2.0)
    assert (output - ref_out).abs().max().item() <= 0.03
    assert (lse_natural - ref_lse).abs().max().item() <= 0.05
    assert _cosine_similarity(output, ref_out) >= 0.99999


@torch.inference_mode()
def test_paged_forward_matches_reference_decode_dense_page128() -> None:
    require_sm120()
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
        q_seqlens=[1, 1],
        cache_seqlens=[200, 384],
        page_size=128,
        q_heads=64,
        kv_heads=4,
        head_dim_qk=128,
        head_dim_vo=128,
        dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
    )
    workspace = _make_workspace(q, k_cache, v_cache, mode="decode")
    workspace.prepare(page_table, cache_seqlens, cu_seqlens_q)
    assert workspace.plan.page_size == 128
    assert workspace.plan.msa_block_sparse is False
    output, lse_base2 = workspace.run(
        q,
        k_cache,
        v_cache,
        output=torch.empty_like(q),
    )
    torch.cuda.synchronize()

    ref_out, ref_lse = paged_attention_reference(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        causal=True,
    )
    lse_natural = lse_base2 * math.log(2.0)
    assert (output - ref_out).abs().max().item() <= 0.03
    assert (lse_natural - ref_lse).abs().max().item() <= 0.05
    assert _cosine_similarity(output, ref_out) >= 0.99999


@torch.inference_mode()
def test_paged_forward_matches_reference_fp8_decode_short_context_batch8() -> None:
    require_sm120()
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
        q_seqlens=[1, 1, 1, 1, 1, 1, 1, 1],
        cache_seqlens=[64, 64, 64, 64, 64, 64, 64, 64],
        dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
    )
    k_fp8, v_fp8, k_descale, v_descale = quantize_paged_kv_cache_e4m3(
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
    )
    workspace = _make_workspace(q, k_fp8, v_fp8, mode="decode")
    workspace.prepare(page_table, cache_seqlens, cu_seqlens_q)
    output, lse_base2 = workspace.run(
        q,
        k_fp8,
        v_fp8,
        output=torch.empty_like(q),
        k_descale=k_descale,
        v_descale=v_descale,
    )
    torch.cuda.synchronize()

    ref_out, ref_lse = paged_attention_reference(
        q,
        k_fp8,
        v_fp8,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        k_descale=k_descale,
        v_descale=v_descale,
        causal=True,
    )
    lse_natural = lse_base2 * math.log(2.0)
    assert (output - ref_out).abs().max().item() <= 0.05
    assert (lse_natural - ref_lse).abs().max().item() <= 0.08
    assert _cosine_similarity(output, ref_out) >= 0.999


@torch.inference_mode()
def test_paged_forward_matches_reference_decode_with_sliding_window_and_sink() -> None:
    require_sm120()
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
        q_seqlens=[1, 1, 1],
        cache_seqlens=[128, 192, 256],
        dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
    )
    window_left = 80
    attention_sink_bias = torch.linspace(-0.2, 0.2, q.shape[1], dtype=torch.float32, device=q.device)
    workspace = _make_workspace(q, k_cache, v_cache, mode="decode")
    workspace.prepare(page_table, cache_seqlens, cu_seqlens_q, window_left=window_left)
    output, lse_base2 = workspace.run(
        q,
        k_cache,
        v_cache,
        output=torch.empty_like(q),
        attention_sink_bias=attention_sink_bias,
    )
    torch.cuda.synchronize()

    ref_out, ref_lse = paged_attention_reference(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        causal=True,
        window_left=window_left,
        attention_sink_bias=attention_sink_bias,
    )
    lse_natural = lse_base2 * math.log(2.0)
    assert (output - ref_out).abs().max().item() <= 0.03
    assert (lse_natural - ref_lse).abs().max().item() <= 0.05
    assert _cosine_similarity(output, ref_out) >= 0.99999


@torch.inference_mode()
def test_paged_forward_matches_reference_decode_mimo_gqa_shape_with_sliding_window_and_sink() -> None:
    require_sm120()
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
        q_seqlens=[1, 1, 1],
        cache_seqlens=[128, 192, 256],
        q_heads=64,
        kv_heads=8,
        head_dim_qk=192,
        head_dim_vo=128,
        dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
    )
    window_left = 80
    attention_sink_bias = torch.linspace(-0.2, 0.2, q.shape[1], dtype=torch.float32, device=q.device)
    workspace = _make_workspace(q, k_cache, v_cache, mode="decode")
    workspace.prepare(page_table, cache_seqlens, cu_seqlens_q, window_left=window_left)
    output, lse_base2 = workspace.run(
        q,
        k_cache,
        v_cache,
        output=torch.empty(q.shape[0], q.shape[1], v_cache.shape[3], dtype=q.dtype, device=q.device),
        attention_sink_bias=attention_sink_bias,
    )
    torch.cuda.synchronize()

    ref_out, ref_lse = paged_attention_reference(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        causal=True,
        window_left=window_left,
        attention_sink_bias=attention_sink_bias,
    )
    lse_natural = lse_base2 * math.log(2.0)
    assert (output - ref_out).abs().max().item() <= 0.02
    assert (lse_natural - ref_lse).abs().max().item() <= 0.03
    assert _cosine_similarity(output, ref_out) >= 0.9999


@torch.inference_mode()
def test_paged_forward_attention_sink_affects_denominator_only() -> None:
    require_sm120()
    q_heads = 8
    kv_heads = 1
    head_dim = 256
    page_size = 64
    q = torch.zeros((1, q_heads, head_dim), dtype=torch.bfloat16, device="cuda")
    k_cache = torch.zeros((1, page_size, kv_heads, head_dim), dtype=torch.bfloat16, device="cuda")
    v_cache = torch.zeros((1, page_size, kv_heads, head_dim), dtype=torch.bfloat16, device="cuda")
    v_cache[:, 0, :, :].fill_(1.0)
    page_table = torch.zeros((1, 1), dtype=torch.int32, device="cuda")
    cache_seqlens = torch.ones((1,), dtype=torch.int32, device="cuda")
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    attention_sink_bias = torch.full((q_heads,), math.log(3.0), dtype=torch.float32, device="cuda")

    workspace = _make_workspace(q, k_cache, v_cache, mode="decode")
    workspace.prepare(page_table, cache_seqlens, cu_seqlens_q)
    output, lse_base2 = workspace.run(
        q,
        k_cache,
        v_cache,
        output=torch.empty_like(q),
        attention_sink_bias=attention_sink_bias,
    )
    torch.cuda.synchronize()

    expected_output = torch.full_like(output, 0.25)
    expected_lse = torch.full((1, q_heads), math.log(4.0), dtype=torch.float32, device="cuda")
    lse_natural = lse_base2 * math.log(2.0)
    assert (output - expected_output).abs().max().item() <= 0.002
    assert (lse_natural - expected_lse).abs().max().item() <= 0.002


@torch.inference_mode()
def test_paged_forward_native_fp8_qkv_matches_reference_fp8_decode_short_context_batch8(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    require_sm120()
    monkeypatch.setenv("B12X_TURBO_ATTN", "1")
    output, ref_out, plan_desc = _run_decode_reference_check(cache_seqlen=64)
    assert plan_desc.endswith(",split")
    assert (output - ref_out).abs().max().item() <= 0.02
    assert _cosine_similarity(output, ref_out) >= 0.995


@torch.inference_mode()
def test_paged_forward_matches_reference_without_split_bf16_extend() -> None:
    require_sm120()
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
        q_seqlens=[6, 5],
        cache_seqlens=[64, 64],
        dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
    )
    workspace = _make_workspace(q, k_cache, v_cache, mode="extend")
    workspace.prepare(
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        disable_split_kv=True,
    )
    output, lse_base2 = workspace.run(q, k_cache, v_cache, output=torch.empty_like(q))
    torch.cuda.synchronize()

    ref_out, ref_lse = paged_attention_reference(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        causal=True,
    )
    lse_natural = lse_base2 * math.log(2.0)
    assert (output - ref_out).abs().max().item() <= 0.03
    assert (lse_natural - ref_lse).abs().max().item() <= 0.05
    assert _cosine_similarity(output, ref_out) >= 0.99999


@torch.inference_mode()
def test_paged_forward_matches_reference_extend_dense_page128() -> None:
    require_sm120()
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
        q_seqlens=[6, 5],
        cache_seqlens=[256, 384],
        page_size=128,
        q_heads=64,
        kv_heads=4,
        head_dim_qk=128,
        head_dim_vo=128,
        dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
    )
    workspace = _make_workspace(q, k_cache, v_cache, mode="extend")
    workspace.prepare(
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        disable_split_kv=True,
    )
    assert workspace.plan.page_size == 128
    assert workspace.plan.msa_block_sparse is False
    output, lse_base2 = workspace.run(q, k_cache, v_cache, output=torch.empty_like(q))
    torch.cuda.synchronize()

    ref_out, ref_lse = paged_attention_reference(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        causal=True,
    )
    lse_natural = lse_base2 * math.log(2.0)
    assert (output - ref_out).abs().max().item() <= 0.03
    assert (lse_natural - ref_lse).abs().max().item() <= 0.05
    assert _cosine_similarity(output, ref_out) >= 0.99999


@torch.inference_mode()
def test_paged_extend_dense_page128_compile_key_uses_fixed_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    require_sm120()
    clear_attention_caches()

    import b12x.attention.paged.api as paged_api

    forward_specs: list[str] = []

    def fake_launch(
        _func,
        *,
        compile_spec,
        compile_args,
        runtime_args,
        compile_kwargs=None,
    ):
        if compile_spec.kernel_id == "attention.paged.forward":
            forward_specs.append(repr(compile_spec))

    monkeypatch.setattr(paged_api, "b12x_launch", fake_launch)

    batch = 4
    page_size = 128
    page_table_width = 4
    num_cache_pages = 64
    max_total_q = 64
    q_heads = 16
    kv_heads = 1
    head_dim = 128
    device = torch.device("cuda")
    dtype = torch.bfloat16

    k_cache = torch.randn(
        num_cache_pages,
        page_size,
        kv_heads,
        head_dim,
        dtype=torch.float32,
        device=device,
    ).to(dtype)
    v_cache = torch.randn_like(k_cache)
    scratch_plan = plan_paged_attention_scratch(
        B12XPagedAttentionScratchCaps(
            device=device,
            mode="extend",
            dtype=dtype,
            kv_dtype=dtype,
            num_q_heads=q_heads,
            num_kv_heads=kv_heads,
            head_dim_qk=head_dim,
            head_dim_vo=head_dim,
            page_size=page_size,
            max_total_q=max_total_q,
            max_batch=batch,
            max_page_table_width=page_table_width,
            max_work_items=128,
            max_partial_rows=0,
            num_cache_pages=num_cache_pages,
        )
    )
    scratch = tuple(
        torch.empty(shape, dtype=scratch_dtype, device=device)
        for shape, scratch_dtype in scratch_plan.shapes_and_dtypes()
    )

    def make_metadata(
        q_lens: list[int],
        cache_lens: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        total_q = sum(q_lens)
        q = torch.randn(total_q, q_heads, head_dim, dtype=dtype, device=device)
        page_table = torch.empty(
            batch, page_table_width, dtype=torch.int32, device=device
        )
        for req_idx, cache_len in enumerate(cache_lens):
            req_pages = math.ceil(cache_len / page_size)
            assert req_pages <= page_table_width
            page_ids = torch.arange(
                req_idx * page_table_width,
                req_idx * page_table_width + req_pages,
                dtype=torch.int32,
                device=device,
            )
            page_table[req_idx, :req_pages] = page_ids
            page_table[req_idx, req_pages:] = page_ids[-1]
        cache_seqlens = torch.tensor(cache_lens, dtype=torch.int32, device=device)
        cu_seqlens_q = torch.tensor(
            [0, *torch.tensor(q_lens, dtype=torch.int32).cumsum(0).tolist()],
            dtype=torch.int32,
            device=device,
        )
        return q, page_table, cache_seqlens, cu_seqlens_q

    for q_lens, cache_lens in (
        ([3, 2, 1, 4], [129, 130, 131, 132]),
        ([16, 8, 4, 1], [400, 401, 402, 403]),
    ):
        q, page_table, cache_seqlens, cu_seqlens_q = make_metadata(
            q_lens,
            cache_lens,
        )
        output = torch.empty_like(q)
        binding = scratch_plan.bind(
            scratch=scratch,
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            output=output,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            active_total_q=q.shape[0],
        )
        paged_attention_forward(binding=binding)

    assert len(forward_specs) == 2
    assert len(set(forward_specs)) == 1


@torch.inference_mode()
def test_paged_forward_matches_reference_extend_with_sliding_window_and_sink() -> None:
    require_sm120()
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
        q_seqlens=[6, 5],
        cache_seqlens=[320, 384],
        dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
    )
    window_left = 96
    attention_sink_bias = torch.linspace(0.1, -0.1, q.shape[1], dtype=torch.float32, device=q.device)
    workspace = _make_workspace(q, k_cache, v_cache, mode="extend")
    workspace.prepare(
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        disable_split_kv=True,
        window_left=window_left,
    )
    output, lse_base2 = workspace.run(
        q,
        k_cache,
        v_cache,
        output=torch.empty_like(q),
        attention_sink_bias=attention_sink_bias,
    )
    torch.cuda.synchronize()

    ref_out, ref_lse = paged_attention_reference(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        causal=True,
        window_left=window_left,
        attention_sink_bias=attention_sink_bias,
    )
    lse_natural = lse_base2 * math.log(2.0)
    assert (output - ref_out).abs().max().item() <= 0.03
    assert (lse_natural - ref_lse).abs().max().item() <= 0.05
    assert _cosine_similarity(output, ref_out) >= 0.99999


@torch.inference_mode()
def test_paged_forward_matches_reference_with_fp8_kv_extend() -> None:
    require_sm120()
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
        q_seqlens=[6, 5],
        cache_seqlens=[2048, 4096],
        dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
    )
    k_fp8, v_fp8, k_descale, v_descale = quantize_paged_kv_cache_e4m3(
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
    )
    workspace = _make_workspace(q, k_fp8, v_fp8, mode="extend")
    workspace.prepare(page_table, cache_seqlens, cu_seqlens_q)
    assert workspace.plan.split_kv is False
    output, lse_base2 = workspace.run(
        q,
        k_fp8,
        v_fp8,
        output=torch.empty_like(q),
        k_descale=k_descale,
        v_descale=v_descale,
    )
    torch.cuda.synchronize()

    ref_out, ref_lse = paged_attention_reference(
        q,
        k_fp8,
        v_fp8,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        k_descale=k_descale,
        v_descale=v_descale,
        causal=True,
    )
    lse_natural = lse_base2 * math.log(2.0)
    assert (output - ref_out).abs().max().item() <= 0.05
    assert (lse_natural - ref_lse).abs().max().item() <= 0.08
    assert _cosine_similarity(output, ref_out) >= 0.999


@torch.inference_mode()
def test_paged_forward_matches_reference_with_split_fp8_decode() -> None:
    require_sm120()
    output, fa2_out, plan_desc = _run_decode_graph_check(cache_seqlen=512)
    assert plan_desc.endswith(",split")
    assert (output - fa2_out).abs().max().item() <= 0.01
    assert _cosine_similarity(output, fa2_out) >= 0.999


@torch.inference_mode()
def test_paged_forward_native_fp8_qkv_matches_reference_with_split_fp8_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    require_sm120()
    monkeypatch.setenv("B12X_TURBO_ATTN", "1")
    output, ref_out, plan_desc = _run_decode_reference_check(cache_seqlen=8192)
    assert plan_desc.endswith(",split")
    assert (output - ref_out).abs().max().item() <= 0.01
    assert _cosine_similarity(output, ref_out) >= 0.995


@torch.inference_mode()
def test_paged_forward_matches_reference_with_bf16_kv_extend() -> None:
    require_sm120()
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
        q_seqlens=[6, 5],
        cache_seqlens=[2048, 4096],
        dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
    )
    workspace = _make_workspace(q, k_cache, v_cache, mode="extend")
    workspace.prepare(page_table, cache_seqlens, cu_seqlens_q)
    assert workspace.plan.split_kv is False
    output, lse_base2 = workspace.run(q, k_cache, v_cache, output=torch.empty_like(q))
    torch.cuda.synchronize()

    ref_out, ref_lse = paged_attention_reference(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        causal=True,
    )
    lse_natural = lse_base2 * math.log(2.0)
    assert (output - ref_out).abs().max().item() <= 0.03
    assert (lse_natural - ref_lse).abs().max().item() <= 0.05
    assert _cosine_similarity(output, ref_out) >= 0.99999
