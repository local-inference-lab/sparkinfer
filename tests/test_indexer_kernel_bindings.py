from __future__ import annotations

import pytest
import torch

import b12x.attention.indexer.extend_kernel as extend_kernel
import b12x.attention.indexer.kernel as paged_kernel


def _paged_tensors():
    q_fp8 = torch.empty((2, 2, 128), dtype=torch.float8_e4m3fn)
    weights = torch.empty((2, 2), dtype=torch.float32)
    index_k_cache = torch.empty((4, 64 * (128 + 4)), dtype=torch.uint8)
    real_page_table = torch.empty((2, 4), dtype=torch.int32)
    seqlens_per_query = torch.empty((2,), dtype=torch.int32)
    active_width = torch.empty((1,), dtype=torch.int32)
    tile_logits = torch.empty((32 * 512,), dtype=torch.float32)
    schedule_metadata = torch.empty((2, 2), dtype=torch.int32)
    return (
        q_fp8,
        weights,
        index_k_cache,
        real_page_table,
        seqlens_per_query,
        active_width,
        tile_logits,
        schedule_metadata,
    )


def _extend_tensors():
    q_fp8 = torch.empty((2, 2, 128), dtype=torch.float8_e4m3fn)
    weights = torch.empty((2, 2), dtype=torch.float32)
    k_quant = torch.empty((64, 128), dtype=torch.float8_e4m3fn)
    k_scale = torch.empty((64,), dtype=torch.float32)
    k_start = torch.empty((2,), dtype=torch.int32)
    k_end = torch.empty((2,), dtype=torch.int32)
    tile_logits = torch.empty((32 * 256,), dtype=torch.float32)
    return q_fp8, weights, k_quant, k_scale, k_start, k_end, tile_logits


def test_paged_logits_kernel_binding_run_uses_binding_argument(monkeypatch) -> None:
    (
        q_fp8,
        weights,
        index_k_cache,
        real_page_table,
        seqlens_per_query,
        active_width,
        _tile_logits,
        schedule_metadata,
    ) = _paged_tensors()
    binding = paged_kernel.build_indexer_paged_logits_kernel_binding(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        real_page_table=real_page_table,
        seqlens_per_query=seqlens_per_query,
        active_width=active_width,
        schedule_metadata=schedule_metadata,
        workspace=object(),
        preinitialize_invalid_logits=False,
    )
    calls = {}

    def fake_run(**kwargs):
        calls.update(kwargs)
        return "logits"

    monkeypatch.setattr(paged_kernel, "run_paged_logits_kernel", fake_run)

    assert binding.run() == "logits"
    assert calls["binding"] is binding


def test_paged_tiled_logits_kernel_binding_supplies_common_call(monkeypatch) -> None:
    (
        q_fp8,
        weights,
        index_k_cache,
        real_page_table,
        seqlens_per_query,
        active_width,
        tile_logits,
        _schedule_metadata,
    ) = _paged_tensors()
    workspace = object()
    phantoms = {"q_bytes": torch.empty((1,), dtype=torch.uint8)}
    binding = paged_kernel.build_indexer_paged_tiled_logits_kernel_binding(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        real_page_table=real_page_table,
        seqlens_per_query=seqlens_per_query,
        active_width=active_width,
        tile_logits=tile_logits,
        contract_phantoms=phantoms,
        workspace=workspace,
        tile_block_q=16,
        preinitialize_tile_logits=False,
    )
    calls = {}

    def fake_common(**kwargs):
        calls.update(kwargs)
        return "tile-logits"

    monkeypatch.setattr(paged_kernel, "_run_paged_tiled_logits_kernel_common", fake_common)

    assert paged_kernel.run_paged_tiled_logits_kernel(binding=binding) == "tile-logits"
    assert calls["q_fp8"] is q_fp8
    assert calls["weights"] is weights
    assert calls["index_k_cache"] is index_k_cache
    assert calls["real_page_table"] is real_page_table
    assert calls["seqlens_per_query"] is seqlens_per_query
    assert calls["active_width"] is active_width
    assert calls["tile_logits"] is tile_logits
    assert calls["contract_phantoms"] is phantoms
    assert calls["workspace"] is workspace
    assert calls["tile_block_q"] == 16
    assert calls["preinitialize_tile_logits"] is False
    assert calls["source_page_offset"] == 0
    assert calls["output_width_tokens"] is None
    assert calls["windowed"] is False


def test_paged_windowed_tiled_logits_kernel_binding_supplies_common_call(monkeypatch) -> None:
    (
        q_fp8,
        weights,
        index_k_cache,
        real_page_table,
        seqlens_per_query,
        active_width,
        tile_logits,
        _schedule_metadata,
    ) = _paged_tensors()
    binding = paged_kernel.build_indexer_paged_windowed_tiled_logits_kernel_binding(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        real_page_table=real_page_table,
        seqlens_per_query=seqlens_per_query,
        active_width=active_width,
        tile_logits=tile_logits,
        source_page_offset=3,
        output_width_tokens=1024,
        stage_runtime_metadata=False,
    )
    calls = {}

    def fake_common(**kwargs):
        calls.update(kwargs)
        return "windowed-tile-logits"

    monkeypatch.setattr(paged_kernel, "_run_paged_tiled_logits_kernel_common", fake_common)

    assert (
        paged_kernel.run_paged_windowed_tiled_logits_kernel(binding=binding)
        == "windowed-tile-logits"
    )
    assert calls["q_fp8"] is q_fp8
    assert calls["tile_logits"] is tile_logits
    assert calls["source_page_offset"] == 3
    assert calls["output_width_tokens"] == 1024
    assert calls["windowed"] is True
    assert calls["stage_runtime_metadata"] is False


def test_paged_logits_kernel_rejects_binding_plus_runtime_tensors() -> None:
    (
        q_fp8,
        weights,
        index_k_cache,
        real_page_table,
        seqlens_per_query,
        *_rest,
    ) = _paged_tensors()
    binding = paged_kernel.build_indexer_paged_logits_kernel_binding(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        real_page_table=real_page_table,
        seqlens_per_query=seqlens_per_query,
    )

    with pytest.raises(ValueError, match="binding owns runtime tensors"):
        paged_kernel.run_paged_logits_kernel(binding=binding, q_fp8=q_fp8)


def test_extend_logits_kernel_binding_run_uses_binding_argument(monkeypatch) -> None:
    q_fp8, weights, k_quant, k_scale, k_start, k_end, tile_logits = _extend_tensors()
    binding = extend_kernel.build_indexer_extend_logits_kernel_binding(
        q_fp8=q_fp8,
        weights=weights,
        k_quant=k_quant,
        k_scale=k_scale,
        k_start=k_start,
        k_end=k_end,
        workspace=object(),
        preinitialize_invalid_logits=False,
        tile_logits=tile_logits,
        tile_k_offset=1,
        tile_num_k_tiles=2,
    )
    calls = {}

    def fake_run(**kwargs):
        calls.update(kwargs)
        return "extend-logits"

    monkeypatch.setattr(extend_kernel, "run_extend_logits_kernel", fake_run)

    assert binding.run() == "extend-logits"
    assert calls["binding"] is binding


def test_extend_logits_kernel_rejects_binding_plus_runtime_tensors() -> None:
    q_fp8, weights, k_quant, k_scale, k_start, k_end, _tile_logits = _extend_tensors()
    binding = extend_kernel.build_indexer_extend_logits_kernel_binding(
        q_fp8=q_fp8,
        weights=weights,
        k_quant=k_quant,
        k_scale=k_scale,
        k_start=k_start,
        k_end=k_end,
    )

    with pytest.raises(ValueError, match="binding owns runtime tensors"):
        extend_kernel.run_extend_logits_kernel(binding=binding, weights=weights)


def test_extend_logits_kernel_without_binding_reports_missing_argument() -> None:
    with pytest.raises(TypeError, match="requires q_fp8 or binding"):
        extend_kernel.run_extend_logits_kernel()
