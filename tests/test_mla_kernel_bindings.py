from __future__ import annotations

import pytest
import torch

import b12x.attention.mla.kernel as mla_kernel
import b12x.attention.mla.kernel_onepass as mla_onepass_kernel
import b12x.attention.mla.split as mla_split


def _sparse_tensors():
    q_all = torch.empty((2, 2, 512), dtype=torch.bfloat16)
    kv_cache = torch.empty((4, 576), dtype=torch.uint8)
    page_table_1 = torch.empty((2, 4), dtype=torch.int32)
    active_token_counts = torch.empty((2,), dtype=torch.int32)
    sm_scale = torch.empty((1,), dtype=torch.float32)
    kv_chunk_size_ptr = torch.empty((1,), dtype=torch.int32)
    num_chunks_ptr = torch.empty((1,), dtype=torch.int32)
    tmp_output = torch.empty((2, 2, 4, 512), dtype=torch.bfloat16)
    tmp_lse = torch.empty((2, 2, 4), dtype=torch.float32)
    output = torch.empty((2, 2, 512), dtype=torch.bfloat16)
    return (
        q_all,
        kv_cache,
        page_table_1,
        active_token_counts,
        sm_scale,
        kv_chunk_size_ptr,
        num_chunks_ptr,
        tmp_output,
        tmp_lse,
        output,
    )


def _compressed_tensors():
    q_all = torch.empty((2, 2, 512), dtype=torch.bfloat16)
    swa_k_cache = torch.empty((4, 1024), dtype=torch.uint8)
    swa_indices = torch.empty((2, 4), dtype=torch.int32)
    swa_lengths = torch.empty((2,), dtype=torch.int32)
    indexed_k_cache = torch.empty((4, 1024), dtype=torch.uint8)
    indexed_indices = torch.empty((2, 4), dtype=torch.int32)
    indexed_lengths = torch.empty((2,), dtype=torch.int32)
    indexed_page_table = torch.empty((2, 4), dtype=torch.int32)
    sm_scale = torch.empty((1,), dtype=torch.float32)
    kv_chunk_size_ptr = torch.empty((1,), dtype=torch.int32)
    num_chunks_ptr = torch.empty((1,), dtype=torch.int32)
    tmp_output = torch.empty((2, 2, 4, 512), dtype=torch.bfloat16)
    tmp_lse = torch.empty((2, 2, 4), dtype=torch.float32)
    return (
        q_all,
        swa_k_cache,
        swa_indices,
        swa_lengths,
        indexed_k_cache,
        indexed_indices,
        indexed_lengths,
        indexed_page_table,
        sm_scale,
        kv_chunk_size_ptr,
        num_chunks_ptr,
        tmp_output,
        tmp_lse,
    )


def test_sparse_mla_kernel_binding_run_uses_binding_argument(monkeypatch) -> None:
    q_all, kv_cache, page_table_1, active_token_counts, sm_scale, *_rest, output = _sparse_tensors()
    binding = mla_kernel.build_sparse_mla_kernel_binding(
        q_all=q_all,
        kv_cache=kv_cache,
        page_table_1=page_table_1,
        active_token_counts=active_token_counts,
        sm_scale=sm_scale,
        output=output,
        workspace=object(),
        identity_page_table=True,
    )
    calls = {}

    def fake_run(**kwargs):
        calls.update(kwargs)

    monkeypatch.setattr(mla_kernel, "run_sparse_mla_kernel", fake_run)

    binding.run()
    assert calls["binding"] is binding


def test_sparse_mla_kernel_rejects_binding_plus_runtime_tensors() -> None:
    q_all, kv_cache, page_table_1, active_token_counts, sm_scale, *_rest, output = _sparse_tensors()
    binding = mla_kernel.build_sparse_mla_kernel_binding(
        q_all=q_all,
        kv_cache=kv_cache,
        page_table_1=page_table_1,
        active_token_counts=active_token_counts,
        sm_scale=sm_scale,
        output=output,
    )

    with pytest.raises(ValueError, match="binding owns runtime tensors"):
        mla_kernel.run_sparse_mla_kernel(binding=binding, q_all=q_all)


def test_sparse_mla_onepass_kernel_binding_run_uses_binding_argument(monkeypatch) -> None:
    q_all, kv_cache, page_table_1, active_token_counts, sm_scale, *_rest, output = _sparse_tensors()
    binding = mla_onepass_kernel.build_sparse_mla_onepass_kernel_binding(
        q_all=q_all,
        kv_cache=kv_cache,
        page_table_1=page_table_1,
        active_token_counts=active_token_counts,
        sm_scale=sm_scale,
        output=output,
        workspace=object(),
    )
    calls = {}

    def fake_run(**kwargs):
        calls.update(kwargs)

    monkeypatch.setattr(mla_onepass_kernel, "run_sparse_mla_kernel", fake_run)

    binding.run()
    assert calls["binding"] is binding


def test_sparse_mla_split_decode_binding_supplies_forward_and_merge(monkeypatch) -> None:
    (
        q_all,
        kv_cache,
        page_table_1,
        active_token_counts,
        sm_scale,
        kv_chunk_size_ptr,
        num_chunks_ptr,
        tmp_output,
        tmp_lse,
        output,
    ) = _sparse_tensors()
    attn_sink = torch.empty((2,), dtype=torch.float32)
    workspace = object()
    binding = mla_split.build_sparse_mla_split_decode_binding(
        q_all=q_all,
        kv_cache=kv_cache,
        page_table_1=page_table_1,
        active_token_counts=active_token_counts,
        sm_scale=sm_scale,
        kv_chunk_size_ptr=kv_chunk_size_ptr,
        num_chunks_ptr=num_chunks_ptr,
        tmp_output=tmp_output,
        tmp_lse=tmp_lse,
        output=output,
        launch_num_chunks=3,
        attn_sink=attn_sink,
        workspace=workspace,
        identity_page_table=True,
    )
    calls = {}

    def fake_forward(**kwargs):
        calls["forward"] = kwargs

    def fake_merge(**kwargs):
        calls["merge"] = kwargs

    monkeypatch.setattr(mla_split, "run_sparse_mla_split_decode_forward", fake_forward)
    monkeypatch.setattr(mla_split, "run_sparse_mla_split_decode_merge", fake_merge)

    mla_split.run_sparse_mla_split_decode(binding=binding)

    assert calls["forward"]["q_all"] is q_all
    assert calls["forward"]["kv_cache"] is kv_cache
    assert calls["forward"]["page_table_1"] is page_table_1
    assert calls["forward"]["active_token_counts"] is active_token_counts
    assert calls["forward"]["sm_scale"] is sm_scale
    assert calls["forward"]["kv_chunk_size_ptr"] is kv_chunk_size_ptr
    assert calls["forward"]["num_chunks_ptr"] is num_chunks_ptr
    assert calls["forward"]["tmp_output"] is tmp_output
    assert calls["forward"]["tmp_lse"] is tmp_lse
    assert calls["forward"]["launch_num_chunks"] == 3
    assert calls["forward"]["workspace"] is workspace
    assert calls["forward"]["identity_page_table"] is True
    assert calls["merge"]["tmp_output"] is tmp_output
    assert calls["merge"]["tmp_lse"] is tmp_lse
    assert calls["merge"]["num_chunks_ptr"] is num_chunks_ptr
    assert calls["merge"]["output"] is output
    assert calls["merge"]["attn_sink"] is attn_sink
    assert calls["merge"]["workspace"] is workspace


def test_sparse_mla_split_forward_binding_run_uses_binding_argument(monkeypatch) -> None:
    (
        q_all,
        kv_cache,
        page_table_1,
        active_token_counts,
        sm_scale,
        kv_chunk_size_ptr,
        num_chunks_ptr,
        tmp_output,
        tmp_lse,
        _output,
    ) = _sparse_tensors()
    binding = mla_split.build_sparse_mla_split_decode_forward_binding(
        q_all=q_all,
        kv_cache=kv_cache,
        page_table_1=page_table_1,
        active_token_counts=active_token_counts,
        sm_scale=sm_scale,
        kv_chunk_size_ptr=kv_chunk_size_ptr,
        num_chunks_ptr=num_chunks_ptr,
        tmp_output=tmp_output,
        tmp_lse=tmp_lse,
        launch_num_chunks=2,
    )
    calls = {}

    def fake_run(**kwargs):
        calls.update(kwargs)

    monkeypatch.setattr(mla_split, "run_sparse_mla_split_decode_forward", fake_run)

    binding.run()
    assert calls["binding"] is binding


def test_compressed_mla_split_forward_binding_run_uses_binding_argument(monkeypatch) -> None:
    (
        q_all,
        swa_k_cache,
        swa_indices,
        swa_lengths,
        indexed_k_cache,
        indexed_indices,
        indexed_lengths,
        indexed_page_table,
        sm_scale,
        kv_chunk_size_ptr,
        num_chunks_ptr,
        tmp_output,
        tmp_lse,
    ) = _compressed_tensors()
    binding = mla_split.build_compressed_mla_split_decode_forward_binding(
        q_all=q_all,
        swa_k_cache=swa_k_cache,
        swa_indices=swa_indices,
        swa_lengths=swa_lengths,
        indexed_k_cache=indexed_k_cache,
        indexed_indices=indexed_indices,
        indexed_lengths=indexed_lengths,
        indexed_page_table=indexed_page_table,
        sm_scale=sm_scale,
        kv_chunk_size_ptr=kv_chunk_size_ptr,
        num_chunks_ptr=num_chunks_ptr,
        tmp_output=tmp_output,
        tmp_lse=tmp_lse,
        launch_num_chunks=2,
        swa_page_size=64,
        swa_page_nbytes=1024,
        indexed_page_size=64,
        indexed_page_nbytes=1024,
        has_indexed=True,
        map_indexed_page_table=False,
        direct_output=False,
        single_tile_chunks=True,
    )
    calls = {}

    def fake_run(**kwargs):
        calls.update(kwargs)

    monkeypatch.setattr(mla_split, "run_compressed_mla_split_decode_forward", fake_run)

    binding.run()
    assert calls["binding"] is binding


def test_sparse_mla_split_forward_rejects_binding_plus_runtime_tensors() -> None:
    (
        q_all,
        kv_cache,
        page_table_1,
        active_token_counts,
        sm_scale,
        kv_chunk_size_ptr,
        num_chunks_ptr,
        tmp_output,
        tmp_lse,
        _output,
    ) = _sparse_tensors()
    binding = mla_split.build_sparse_mla_split_decode_forward_binding(
        q_all=q_all,
        kv_cache=kv_cache,
        page_table_1=page_table_1,
        active_token_counts=active_token_counts,
        sm_scale=sm_scale,
        kv_chunk_size_ptr=kv_chunk_size_ptr,
        num_chunks_ptr=num_chunks_ptr,
        tmp_output=tmp_output,
        tmp_lse=tmp_lse,
        launch_num_chunks=2,
    )

    with pytest.raises(ValueError, match="binding owns runtime tensors"):
        mla_split.run_sparse_mla_split_decode_forward(binding=binding, tmp_output=tmp_output)


def test_sparse_mla_split_decode_without_binding_reports_missing_argument() -> None:
    with pytest.raises(TypeError, match="requires q_all or binding"):
        mla_split.run_sparse_mla_split_decode()
