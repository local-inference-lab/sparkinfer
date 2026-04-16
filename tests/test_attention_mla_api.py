from __future__ import annotations

import pytest
import torch

from b12x.integration.mla import (
    MLASparseDecodeMetadata,
    MLASparseExtendMetadata,
    MLAWorkspace,
    sparse_mla_decode_forward,
    sparse_mla_extend_forward,
)


def _make_workspace(*, mode: str, topk: int = 4) -> MLAWorkspace:
    return MLAWorkspace.for_fixed_capacity(
        mode=mode,
        device="cpu",
        dtype=torch.bfloat16,
        kv_dtype=torch.uint8,
        num_q_heads=8,
        head_dim=256,
        v_head_dim=256,
        topk=topk,
        max_total_q=8,
        max_batch=4,
    )


def test_sparse_mla_decode_keeps_query_head_shape(monkeypatch) -> None:
    workspace = _make_workspace(mode="decode")
    captured: dict[str, torch.Tensor | float | int] = {}

    def fake_sparse_mla_reference(
        *,
        q_all,
        kv_cache,
        page_table_1,
        active_token_counts=None,
        sm_scale,
        v_head_dim,
    ):
        captured["q"] = q_all
        captured["page_table_1"] = page_table_1
        captured["active_token_counts"] = active_token_counts
        captured["kv_cache"] = kv_cache
        captured["sm_scale"] = sm_scale
        captured["d_v"] = v_head_dim
        return q_all[:, :, :v_head_dim].clone()

    monkeypatch.setattr(
        "b12x.attention.mla.api.sparse_mla_reference",
        fake_sparse_mla_reference,
    )

    q_all = torch.ones((2, 8, 256), dtype=torch.bfloat16)
    kv_cache = torch.zeros((16, 1, 656), dtype=torch.uint8)
    page_table_1 = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.int32)
    cache_seqlens = torch.tensor([8, 8], dtype=torch.int32)
    metadata = MLASparseDecodeMetadata(
        page_table_1=page_table_1,
        cache_seqlens_int32=cache_seqlens,
        nsa_cache_seqlens_int32=cache_seqlens,
        max_seq_len_k=8,
    )

    output = sparse_mla_decode_forward(
        q_all=q_all,
        kv_cache=kv_cache,
        metadata=metadata,
        workspace=workspace,
        sm_scale=0.5,
        v_head_dim=256,
    )

    assert output.shape == (2, 8, 256)
    assert captured["q"].shape == (2, 8, 256)
    assert captured["page_table_1"].shape == (2, 4)
    assert captured["sm_scale"] == 0.5
    assert captured["d_v"] == 256
    assert workspace.page_table_1 is not page_table_1
    assert torch.equal(workspace.page_table_1, page_table_1)


def test_sparse_mla_extend_uses_bound_metadata(monkeypatch) -> None:
    workspace = _make_workspace(mode="extend", topk=6)

    def fake_sparse_mla_reference(
        *,
        q_all,
        kv_cache,
        page_table_1,
        active_token_counts=None,
        sm_scale,
        v_head_dim,
    ):
        del kv_cache, page_table_1, active_token_counts, sm_scale, v_head_dim
        return q_all[:, :8, :].clone()

    monkeypatch.setattr(
        "b12x.attention.mla.api.sparse_mla_reference",
        fake_sparse_mla_reference,
    )

    q_all = torch.ones((3, 8, 256), dtype=torch.bfloat16)
    kv_cache = torch.zeros((32, 1, 656), dtype=torch.uint8)
    page_table_1 = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10, 11],
            [12, 13, 14, 15, 16, 17],
        ],
        dtype=torch.int32,
    )
    cache_seqlens = torch.tensor([12, 12, 12], dtype=torch.int32)
    nsa_cu = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    metadata = MLASparseExtendMetadata(
        selected_token_offsets=page_table_1,
        cache_seqlens_int32=cache_seqlens,
        nsa_cache_seqlens_int32=cache_seqlens,
        nsa_cu_seqlens_q=nsa_cu,
        nsa_cu_seqlens_k=nsa_cu,
        max_seq_len_q=1,
        max_seq_len_k=12,
        mode="draft_extend",
    )

    output = sparse_mla_extend_forward(
        q_all=q_all,
        kv_cache=kv_cache,
        metadata=metadata,
        workspace=workspace,
        sm_scale=1.0,
        v_head_dim=256,
    )

    assert output.shape == (3, 8, 256)
    assert workspace.cache_seqlens_int32 is not cache_seqlens
    assert workspace.nsa_cache_seqlens_int32 is not cache_seqlens
    assert torch.equal(workspace.cache_seqlens_int32, cache_seqlens)
    assert torch.equal(workspace.nsa_cache_seqlens_int32, cache_seqlens)


def test_mla_verify_workspace_allocates_split_buffers() -> None:
    workspace = _make_workspace(mode="verify", topk=6)

    assert workspace.mode == "verify"
    assert workspace.tmp_output is not None
    assert workspace.tmp_lse is not None


def test_sparse_mla_verify_prefers_split_path(monkeypatch) -> None:
    workspace = _make_workspace(mode="verify", topk=2048)
    captured: dict[str, object] = {}

    def fake_select_split(**kwargs):
        del kwargs
        from b12x.attention.mla.split import SparseMLASplitDecodeConfig

        return SparseMLASplitDecodeConfig(chunk_size=32, num_chunks=64)

    def fake_run_split_decode(**kwargs):
        captured["run_split"] = True
        output = kwargs["output"]
        output.zero_()

    def fail_run_sparse_mla_kernel(**kwargs):
        raise AssertionError("verify path should not use generic sparse MLA kernel")

    monkeypatch.setattr(
        "b12x.attention.mla.api.select_sparse_mla_split_decode_config",
        fake_select_split,
    )
    monkeypatch.setattr(
        "b12x.attention.mla.api.run_sparse_mla_split_decode",
        fake_run_split_decode,
    )
    monkeypatch.setattr(
        "b12x.attention.mla.api.run_sparse_mla_kernel",
        fail_run_sparse_mla_kernel,
    )

    q_all = torch.ones((5, 8, 256), dtype=torch.bfloat16)
    kv_cache = torch.zeros((32, 1, 656), dtype=torch.uint8)
    page_table_1 = torch.zeros((5, 2048), dtype=torch.int32)
    cache_seqlens = torch.full((1,), 12, dtype=torch.int32)
    nsa_cache_seqlens = torch.full((5,), 12, dtype=torch.int32)
    nsa_cu = torch.tensor([0, 5], dtype=torch.int32)
    metadata = MLASparseExtendMetadata(
        selected_token_offsets=page_table_1,
        cache_seqlens_int32=cache_seqlens,
        nsa_cache_seqlens_int32=nsa_cache_seqlens,
        nsa_cu_seqlens_q=nsa_cu,
        nsa_cu_seqlens_k=nsa_cu,
        max_seq_len_q=5,
        max_seq_len_k=12,
        mode="target_verify",
    )

    output = sparse_mla_extend_forward(
        q_all=q_all,
        kv_cache=kv_cache,
        metadata=metadata,
        workspace=workspace,
        sm_scale=1.0,
        v_head_dim=256,
    )

    assert output.shape == (5, 8, 256)
    assert captured["run_split"] is True


def test_sparse_mla_extend_prefers_split_path(monkeypatch) -> None:
    workspace = _make_workspace(mode="extend", topk=2048)
    captured: dict[str, object] = {}

    def fake_select_split(**kwargs):
        del kwargs
        from b12x.attention.mla.split import SparseMLASplitDecodeConfig

        return SparseMLASplitDecodeConfig(chunk_size=64, num_chunks=32)

    def fake_run_split_decode(**kwargs):
        captured["active_token_counts"] = kwargs["active_token_counts"].clone()
        output = kwargs["output"]
        output.zero_()

    def fail_run_sparse_mla_kernel(**kwargs):
        raise AssertionError("extend split path should not use generic sparse MLA kernel")

    monkeypatch.setattr(
        "b12x.attention.mla.api.select_sparse_mla_split_decode_config",
        fake_select_split,
    )
    monkeypatch.setattr(
        "b12x.attention.mla.api.run_sparse_mla_split_decode",
        fake_run_split_decode,
    )
    monkeypatch.setattr(
        "b12x.attention.mla.api.run_sparse_mla_kernel",
        fail_run_sparse_mla_kernel,
    )

    q_all = torch.ones((5, 8, 256), dtype=torch.bfloat16)
    kv_cache = torch.zeros((32, 1, 656), dtype=torch.uint8)
    page_table_1 = torch.zeros((5, 2048), dtype=torch.int32)
    cache_seqlens = torch.full((1,), 12, dtype=torch.int32)
    nsa_cache_seqlens = torch.tensor([1537, 1024, 257, 64, 0], dtype=torch.int32)
    nsa_cu = torch.tensor([0, 5], dtype=torch.int32)
    metadata = MLASparseExtendMetadata(
        selected_token_offsets=page_table_1,
        cache_seqlens_int32=cache_seqlens,
        nsa_cache_seqlens_int32=nsa_cache_seqlens,
        nsa_cu_seqlens_q=nsa_cu,
        nsa_cu_seqlens_k=nsa_cu,
        max_seq_len_q=5,
        max_seq_len_k=12,
        mode="extend",
    )

    output = sparse_mla_extend_forward(
        q_all=q_all,
        kv_cache=kv_cache,
        metadata=metadata,
        workspace=workspace,
        sm_scale=1.0,
        v_head_dim=256,
    )

    assert output.shape == (5, 8, 256)
    assert torch.equal(captured["active_token_counts"], nsa_cache_seqlens)


def test_sparse_mla_extend_passes_active_token_counts_to_kernel(monkeypatch) -> None:
    workspace = _make_workspace(mode="extend", topk=6)
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "b12x.attention.mla.api.select_sparse_mla_split_decode_config",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "b12x.attention.mla.api.supports_sparse_mla_kernel",
        lambda **kwargs: True,
    )

    def fake_run_sparse_mla_kernel(**kwargs):
        captured["active_token_counts"] = kwargs["active_token_counts"].clone()
        kwargs["output"].zero_()

    monkeypatch.setattr(
        "b12x.attention.mla.api.run_sparse_mla_kernel",
        fake_run_sparse_mla_kernel,
    )

    q_all = torch.ones((3, 8, 256), dtype=torch.bfloat16)
    kv_cache = torch.zeros((32, 1, 656), dtype=torch.uint8)
    page_table_1 = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10, 11],
            [12, 13, 14, 15, 16, 17],
        ],
        dtype=torch.int32,
    )
    cache_seqlens = torch.tensor([12], dtype=torch.int32)
    nsa_cache_seqlens = torch.tensor([6, 4, 2], dtype=torch.int32)
    nsa_cu = torch.tensor([0, 3], dtype=torch.int32)
    metadata = MLASparseExtendMetadata(
        selected_token_offsets=page_table_1,
        cache_seqlens_int32=cache_seqlens,
        nsa_cache_seqlens_int32=nsa_cache_seqlens,
        nsa_cu_seqlens_q=nsa_cu,
        nsa_cu_seqlens_k=nsa_cu,
        max_seq_len_q=3,
        max_seq_len_k=12,
        mode="extend",
    )

    output = sparse_mla_extend_forward(
        q_all=q_all,
        kv_cache=kv_cache,
        metadata=metadata,
        workspace=workspace,
        sm_scale=1.0,
        v_head_dim=256,
    )

    assert output.shape == (3, 8, 256)
    assert torch.equal(captured["active_token_counts"], nsa_cache_seqlens)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for graph capture coverage")
def test_sparse_mla_extend_split_graph_capture_uses_capture_safe_scalar_updates(monkeypatch) -> None:
    workspace = MLAWorkspace.for_fixed_capacity(
        mode="extend",
        device="cuda",
        dtype=torch.bfloat16,
        kv_dtype=torch.uint8,
        num_q_heads=8,
        head_dim=256,
        v_head_dim=256,
        topk=64,
        max_total_q=4,
        max_batch=1,
        use_cuda_graph=True,
    )

    def fake_select_split(**kwargs):
        del kwargs
        from b12x.attention.mla.split import SparseMLASplitDecodeConfig

        return SparseMLASplitDecodeConfig(chunk_size=32, num_chunks=2)

    def fake_run_split_decode(**kwargs):
        kwargs["output"].zero_()

    monkeypatch.setattr(
        "b12x.attention.mla.api.select_sparse_mla_split_decode_config",
        fake_select_split,
    )
    monkeypatch.setattr(
        "b12x.attention.mla.api.run_sparse_mla_split_decode",
        fake_run_split_decode,
    )

    q_all = torch.ones((4, 8, 256), dtype=torch.bfloat16, device="cuda")
    kv_cache = torch.zeros((32, 1, 656), dtype=torch.uint8, device="cuda")
    page_table_1 = torch.arange(64, dtype=torch.int32, device="cuda").unsqueeze(0).expand(4, -1).clone()
    cache_seqlens = torch.tensor([64], dtype=torch.int32, device="cuda")
    nsa_cache_seqlens = torch.tensor([64, 48, 32, 16], dtype=torch.int32, device="cuda")
    nsa_cu = torch.tensor([0, 4], dtype=torch.int32, device="cuda")
    metadata = MLASparseExtendMetadata(
        selected_token_offsets=page_table_1,
        cache_seqlens_int32=cache_seqlens,
        nsa_cache_seqlens_int32=nsa_cache_seqlens,
        nsa_cu_seqlens_q=nsa_cu,
        nsa_cu_seqlens_k=nsa_cu,
        max_seq_len_q=4,
        max_seq_len_k=64,
        mode="extend",
    )

    captured_out = None

    def run() -> None:
        nonlocal captured_out
        captured_out = sparse_mla_extend_forward(
            q_all=q_all,
            kv_cache=kv_cache,
            metadata=metadata,
            workspace=workspace,
            sm_scale=0.5,
            v_head_dim=256,
        )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    torch.cuda.synchronize()

    graph.replay()
    torch.cuda.synchronize()

    assert captured_out is not None
    assert captured_out.shape == (4, 8, 256)
    assert workspace.kv_chunk_size_ptr is not None
    assert workspace.num_chunks_ptr is not None
    assert int(workspace.kv_chunk_size_ptr[0].item()) == 32
    assert int(workspace.num_chunks_ptr[0].item()) == 2
    assert workspace.sm_scale_tensor is not None
    assert float(workspace.sm_scale_tensor[0].item()) == pytest.approx(0.5)
    assert torch.count_nonzero(captured_out).item() == 0


def test_mla_workspace_graph_mode_copies_runtime_metadata() -> None:
    workspace = MLAWorkspace.for_contract(
        mode="decode",
        device="cpu",
        dtype=torch.bfloat16,
        kv_dtype=torch.uint8,
        num_q_heads=8,
        head_dim=256,
        v_head_dim=256,
        topk=4,
        max_total_q=8,
        max_batch=4,
        use_cuda_graph=True,
    )

    page_table_1 = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.int32)
    cache_seqlens = torch.tensor([8, 8], dtype=torch.int32)
    nsa_cache_seqlens = torch.tensor([4, 4], dtype=torch.int32)
    workspace.prepare_decode(page_table_1, cache_seqlens, nsa_cache_seqlens)

    assert workspace.page_table_1 is not page_table_1
    assert workspace.cache_seqlens_int32 is not cache_seqlens
    assert workspace.nsa_cache_seqlens_int32 is not nsa_cache_seqlens
    assert torch.equal(workspace.page_table_1, page_table_1)
    assert torch.equal(workspace.cache_seqlens_int32, cache_seqlens)
    assert torch.equal(workspace.nsa_cache_seqlens_int32, nsa_cache_seqlens)


def test_mla_decode_workspace_allocates_split_buffers_and_chunk_scalars() -> None:
    workspace = MLAWorkspace.for_fixed_capacity(
        mode="decode",
        device="cpu",
        dtype=torch.bfloat16,
        kv_dtype=torch.uint8,
        num_q_heads=8,
        head_dim=256,
        v_head_dim=256,
        topk=2048,
        max_total_q=8,
        max_batch=4,
    )

    assert workspace.tmp_output is not None
    assert workspace.tmp_output.shape == (8, 8, workspace.max_chunks_per_row, 256)
    assert workspace.tmp_lse is not None
    assert workspace.tmp_lse.shape == (8, 8, workspace.max_chunks_per_row)
    workspace.set_decode_chunk_config(kv_chunk_size=256, num_chunks=8)
    assert workspace.kv_chunk_size_ptr is not None
    assert workspace.num_chunks_ptr is not None
    assert int(workspace.kv_chunk_size_ptr[0].item()) == 256
    assert int(workspace.num_chunks_ptr[0].item()) == 8


def test_mla_workspace_enforces_capacity_limits() -> None:
    workspace = _make_workspace(mode="decode", topk=4)
    with torch.no_grad():
        too_wide = torch.zeros((2, 5), dtype=torch.int32)
        cache_seqlens = torch.zeros((2,), dtype=torch.int32)
        try:
            workspace.prepare_decode(too_wide, cache_seqlens, cache_seqlens)
        except ValueError as exc:
            assert "topk capacity" in str(exc)
        else:
            raise AssertionError("expected capacity validation to fail")
