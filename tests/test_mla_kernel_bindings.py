from __future__ import annotations

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

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


def test_b12x_mla_custom_ops_have_fake_dispatch() -> None:
    # Import modules for registration side effects.
    __import__("b12x.attention.mla.unified_sm120.launch")
    __import__("b12x.attention.mla.unified_sm120.prefill")
    # prefill_mg registers the single-cache MG op + the new dual-cache MG op
    # (unified_sm120_prefill_mg_dual). prefill.py imports it lazily, so import it
    # explicitly here for the binding/fake-dispatch coverage.
    __import__("b12x.attention.mla.unified_sm120.prefill_mg")

    with FakeTensorMode():
        q_all = torch.empty((2, 2, 512), dtype=torch.bfloat16)
        cache = torch.empty((4, 1024), dtype=torch.uint8)
        indices = torch.empty((2, 4), dtype=torch.int32)
        lengths = torch.empty((2,), dtype=torch.int32)
        scalar_i32 = torch.empty((1,), dtype=torch.int32)
        sm_scale_t = torch.empty((1,), dtype=torch.float32)
        tmp_output = torch.empty((2, 2, 4, 512), dtype=torch.bfloat16)
        tmp_lse = torch.empty((2, 2, 4), dtype=torch.float32)
        attn_sink = torch.empty((2,), dtype=torch.float32)
        output = torch.empty((2, 2, 512), dtype=torch.bfloat16)

        torch.ops.b12x.compressed_mla_split_decode_forward(
            q_all,
            cache,
            indices,
            lengths,
            cache,
            indices,
            lengths,
            indices,
            sm_scale_t,
            scalar_i32,
            scalar_i32,
            tmp_output,
            tmp_lse,
            attn_sink,
            2,
            64,
            1024,
            64,
            1024,
            True,
            True,
            False,
            False,
            True,
            False,
        )
        torch.ops.b12x.sparse_mla_split_decode_merge(
            tmp_output,
            tmp_lse,
            scalar_i32,
            output,
            attn_sink,
            tmp_output,
            tmp_lse,
            output,
            True,
        )

        mid_output = torch.empty((2, 2, 2, 512), dtype=torch.bfloat16)
        mid_lse = torch.empty((2, 2, 2), dtype=torch.float32)
        torch.ops.b12x.unified_sm120_decode_grid(
            q_all,
            cache,
            indices,
            mid_output,
            mid_lse,
            lengths,
            cache,
            indices,
            lengths,
            0.1,
            0,
            0,
            0,
            64,
            4,
            4,
            1,
            2,
            1,
            1024,
            64,
            1024,
            1,
            2,
            0,
            True,
            False,
        )

        prefill_lse = torch.empty((2, 2), dtype=torch.float32)
        # The single-cache decode-reuse prefill op was REMOVED (no fallback kernel
        # in prefill.py); the only prefill op is the MG dual-cache op below.
        # DUAL-CACHE MG prefill op (the new op DSV4 has_extra routes through).
        torch.ops.b12x.unified_sm120_prefill_mg_dual(
            q_all,        # q
            cache,        # kv_flat (MAIN)
            indices,      # topk_indices
            lengths,      # topk_length
            attn_sink,    # attn_sink_t
            output,       # output
            prefill_lse,  # lse_out
            cache,        # extra_kv_flat
            indices,      # extra_indices_t
            lengths,      # extra_len_t
            0.1,          # sm_scale
            64,           # page_block_size
            4,            # topk
            2,            # num_tiles
            1024,         # stride_kv_block
            True,         # has_sink
            1,            # compute_mode (BF16)
            2,            # mg_n_hg
            0,            # model_type (DSV4)
            0,            # scale_format
            4,            # extra_topk
            1,            # num_main_tiles
            2,            # pbs_extra
            1024,         # stride_extra_kv_block
            True,         # row_xor
        )


def test_unified_sm120_prefill_dual_odd_multiple_heads_splits_to_mg(monkeypatch) -> None:
    # DSV4 dual-cache heads=80 is a paired 64-head MG prefix plus one 16-head
    # single-group tail. This is Python dispatch coverage only; the focused CUDA
    # numerics live in the SM120 test suite.
    __import__("b12x.attention.mla.unified_sm120.prefill")
    import b12x.attention.mla.unified_sm120.prefill_mg as prefill_mg
    from b12x.attention.mla.unified_sm120.prefill import run_unified_prefill

    calls = []

    def fake_run_unified_prefill_mg(**kwargs):
        calls.append(kwargs)
        return kwargs["output"], kwargs["lse_out"]

    monkeypatch.setattr(prefill_mg, "run_unified_prefill_mg", fake_run_unified_prefill_mg)

    topk = 128
    q = torch.empty((2, 80, 512), dtype=torch.bfloat16)
    kv_cache = torch.empty((4, 1024), dtype=torch.uint8)
    topk_indices = torch.zeros((2, topk), dtype=torch.int32)
    extra_kv_cache = torch.empty((4, 1024), dtype=torch.uint8)
    extra_indices = torch.zeros((2, 128), dtype=torch.int32)

    output, lse = run_unified_prefill(
        q=q,
        kv_cache=kv_cache,
        topk_indices=topk_indices,
        sm_scale=0.1,
        page_block_size=64,
        extra_kv_cache=extra_kv_cache,
        extra_indices=extra_indices,
        extra_page_block_size=2,
    )

    assert len(calls) == 2
    assert output.shape == (2, 80, 512)
    assert lse.shape == (2, 80)
    assert calls[0]["mg_n_hg"] == 2
    assert calls[0]["active_heads"] == 64
    assert calls[0]["head_offset"] == 0
    assert calls[1]["mg_n_hg"] == 1
    assert calls[1]["active_heads"] == 16
    assert calls[1]["head_offset"] == 64
    assert calls[0]["output"] is output
    assert calls[1]["output"] is output
    assert calls[0]["lse_out"] is lse
    assert calls[1]["lse_out"] is lse


def test_unified_sm120_prefill_dual_non_eligible_raises() -> None:
    # DSV4 dual-cache prefill is MG-only (topk==128, heads divisible by 16);
    # everything else RAISEs (the decode-reuse has_extra fallback was removed).
    # topk != 128 (here topk == 64) is non-eligible -> ValueError, raised in the
    # Python dispatch BEFORE any kernel launch (so this runs on CPU tensors).
    __import__("b12x.attention.mla.unified_sm120.prefill")
    from b12x.attention.mla.unified_sm120.prefill import run_unified_prefill

    topk = 64  # != 128 -> non-eligible dual
    q = torch.empty((2, 32, 512), dtype=torch.bfloat16)
    kv_cache = torch.empty((4, 1024), dtype=torch.uint8)
    topk_indices = torch.zeros((2, topk), dtype=torch.int32)
    extra_kv_cache = torch.empty((4, 1024), dtype=torch.uint8)
    extra_indices = torch.zeros((2, 64), dtype=torch.int32)

    with pytest.raises(ValueError, match="requires MG dispatch"):
        run_unified_prefill(
            q=q,
            kv_cache=kv_cache,
            topk_indices=topk_indices,
            sm_scale=0.1,
            page_block_size=64,
            extra_kv_cache=extra_kv_cache,
            extra_indices=extra_indices,
            extra_page_block_size=2,
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
        scratch=object(),
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
        scratch=object(),
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
    scratch = object()
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
        scratch=scratch,
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
    assert calls["forward"]["workspace"] is scratch
    assert calls["forward"]["identity_page_table"] is True
    assert calls["merge"]["tmp_output"] is tmp_output
    assert calls["merge"]["tmp_lse"] is tmp_lse
    assert calls["merge"]["num_chunks_ptr"] is num_chunks_ptr
    assert calls["merge"]["output"] is output
    assert calls["merge"]["attn_sink"] is attn_sink
    assert calls["merge"]["workspace"] is scratch


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
