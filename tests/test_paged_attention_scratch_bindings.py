from __future__ import annotations

import pytest
import torch

import b12x.attention.paged.api as paged_api
from b12x.integration.attention import (
    B12XPagedAttentionBinding,
    B12XPagedAttentionScratchCaps,
    PagedAttentionWorkspace,
    plan_paged_attention_scratch,
)


def _caps() -> B12XPagedAttentionScratchCaps:
    return B12XPagedAttentionScratchCaps(
        device="cpu",
        mode="decode",
        dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
        num_q_heads=2,
        num_kv_heads=1,
        head_dim_qk=16,
        head_dim_vo=16,
        page_size=4,
        max_total_q=2,
        max_batch=2,
        max_page_table_width=3,
        max_work_items=4,
        max_partial_rows=4,
        num_cache_pages=8,
    )


def _workspace() -> PagedAttentionWorkspace:
    caps = _caps()
    return PagedAttentionWorkspace.for_fixed_capacity(
        mode=caps.mode,
        device=caps.device,
        dtype=caps.dtype,
        kv_dtype=caps.kv_dtype,
        num_q_heads=caps.num_q_heads,
        num_kv_heads=caps.num_kv_heads,
        head_dim_qk=caps.head_dim_qk,
        head_dim_vo=caps.head_dim_vo,
        page_size=caps.page_size,
        max_total_q=caps.max_total_q,
        max_batch=caps.max_batch,
        max_page_table_width=caps.max_page_table_width,
        max_work_items=caps.max_work_items,
        max_partial_rows=caps.max_partial_rows,
        num_cache_pages=caps.num_cache_pages,
    )


def _runtime_tensors():
    q = torch.empty((2, 2, 16), dtype=torch.bfloat16)
    k_cache = torch.empty((8, 4, 1, 16), dtype=torch.bfloat16)
    v_cache = torch.empty((8, 4, 1, 16), dtype=torch.bfloat16)
    output = torch.empty((2, 2, 16), dtype=torch.bfloat16)
    page_table = torch.zeros((2, 3), dtype=torch.int32)
    cache_seqlens = torch.ones((2,), dtype=torch.int32)
    cu_seqlens_q = torch.arange(3, dtype=torch.int32)
    return q, k_cache, v_cache, output, page_table, cache_seqlens, cu_seqlens_q


def test_paged_attention_scratch_plan_exposes_one_opaque_arena_spec() -> None:
    plan = plan_paged_attention_scratch(_caps())

    specs = plan.scratch_specs()
    assert len(specs) == 1
    assert specs[0].name == "paged_attention.arena"
    assert specs[0].dtype == torch.uint8
    assert specs[0].shape == plan.shapes_and_dtypes()[0][0]
    assert specs[0].nbytes == specs[0].shape[0]
    assert plan.contract.max_total_q == 2


def test_workspace_bind_paged_attention_returns_common_binding_type(monkeypatch) -> None:
    workspace = _workspace()
    q, k_cache, v_cache, output, page_table, cache_seqlens, cu_seqlens_q = _runtime_tensors()
    calls = {}

    def fake_prepare(self, page_table_arg, cache_seqlens_arg, cu_seqlens_q_arg, **kwargs):
        calls["page_table"] = page_table_arg
        calls["cache_seqlens"] = cache_seqlens_arg
        calls["cu_seqlens_q"] = cu_seqlens_q_arg
        calls["kwargs"] = kwargs
        self.page_table = page_table_arg
        self.cache_seqlens = cache_seqlens_arg
        self.cu_seqlens_q = cu_seqlens_q_arg
        self._plan = object()
        return self

    monkeypatch.setattr(PagedAttentionWorkspace, "prepare", fake_prepare)

    binding = workspace.bind_paged_attention(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        output=output,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        window_left=7,
        active_total_q=2,
    )

    assert isinstance(binding, B12XPagedAttentionBinding)
    assert binding.workspace is workspace
    assert binding.q is q
    assert binding.output is output
    assert calls["page_table"] is page_table
    assert calls["kwargs"]["window_left"] == 7
    assert calls["kwargs"]["active_total_q"] == 2


def test_paged_attention_binding_run_uses_function_binding_argument(monkeypatch) -> None:
    workspace = _workspace()
    q, k_cache, v_cache, output, *_ = _runtime_tensors()
    binding = B12XPagedAttentionBinding(
        workspace=workspace,
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        output=output,
    )
    calls = {}

    def fake_forward(**kwargs):
        calls.update(kwargs)
        return "out", "lse"

    monkeypatch.setattr(paged_api, "paged_attention_forward", fake_forward)

    assert binding.run() == ("out", "lse")
    assert calls["binding"] is binding


def test_paged_attention_forward_rejects_binding_plus_runtime_tensors() -> None:
    workspace = _workspace()
    q, k_cache, v_cache, output, *_ = _runtime_tensors()
    binding = B12XPagedAttentionBinding(
        workspace=workspace,
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        output=output,
    )

    with pytest.raises(ValueError, match="binding owns runtime tensors"):
        paged_api.paged_attention_forward(binding=binding, q=q)
