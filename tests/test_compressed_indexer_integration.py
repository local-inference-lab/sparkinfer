from __future__ import annotations

import pytest
import torch

from b12x.attention.workspace import (
    B12XAttentionArena,
    B12XAttentionArenaCaps,
    B12XAttentionWorkspaceContract,
)
from b12x.integration import (
    B12XAttentionWorkspace,
    clear_indexer_caches,
    pack_compressed_index_k_cache_reference,
    compressed_index_decode_dense_topk_fp8,
    compressed_index_decode_logits_fp8,
    compressed_index_decode_supertile_topk_fp8,
    compressed_index_logits_reference,
    prepare_compressed_indexer_metadata,
    resolve_replicated_num_q_heads,
)


def _make_real_page_table(
    *,
    page_starts: list[int],
    seqlens: list[int],
    width_blocks: int,
    device: torch.device,
) -> torch.Tensor:
    real_page_table = torch.full(
        (len(seqlens), width_blocks),
        -1,
        dtype=torch.int32,
        device=device,
    )
    for row_idx, (page_start, seq_len) in enumerate(zip(page_starts, seqlens, strict=True)):
        block_count = (int(seq_len) + 63) // 64
        if block_count:
            real_page_table[row_idx, :block_count] = torch.arange(
                page_start,
                page_start + block_count,
                dtype=torch.int32,
                device=device,
            )
    return real_page_table.contiguous()


def _rand_fp8_q(
    shape: tuple[int, int, int],
    *,
    gen: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    return (
        torch.randn(shape, generator=gen, dtype=torch.float32).to(device=device) / 2
    ).to(torch.float8_e4m3fn)


def _expected_compressed_index_topk(
    *,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    index_k_cache: torch.Tensor,
    real_page_table: torch.Tensor,
    seqlens: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    logits = compressed_index_logits_reference(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        real_page_table=real_page_table,
        query_row_to_batch=torch.arange(
            q_fp8.shape[0],
            dtype=torch.int32,
            device=q_fp8.device,
        ),
        seqlens_per_query=seqlens,
    )
    raw = torch.topk(logits, k=topk, dim=1, largest=True, sorted=False).indices.to(
        torch.int32
    )
    return raw


def test_resolve_replicated_num_q_heads_for_tensor_parallel() -> None:
    assert resolve_replicated_num_q_heads(global_num_q_heads=64, tensor_parallel_size=2) == 64
    assert resolve_replicated_num_q_heads(global_num_q_heads=64, tensor_parallel_size=1) == 64
    with pytest.raises(ValueError, match="must be positive"):
        resolve_replicated_num_q_heads(global_num_q_heads=0, tensor_parallel_size=2)
    with pytest.raises(ValueError, match="tensor_parallel_size must be positive"):
        resolve_replicated_num_q_heads(global_num_q_heads=64, tensor_parallel_size=0)


def test_compressed_index_decode_logits_fp8_hard_fails_on_cpu() -> None:
    device = torch.device("cpu")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(91_001)

    rows = 3
    num_heads = 4
    page_starts = [1, 4, 8]
    seqlens = torch.tensor([65, 128, 150], dtype=torch.int32, device=device)
    width_blocks = 3
    real_page_table = _make_real_page_table(
        page_starts=page_starts,
        seqlens=seqlens.tolist(),
        width_blocks=width_blocks,
        device=device,
    )
    q_fp8 = _rand_fp8_q((rows, num_heads, 128), gen=gen, device=device)
    weights = torch.randn((rows, num_heads), generator=gen, dtype=torch.float32, device=device)
    index_k_cache = pack_compressed_index_k_cache_reference(
        torch.randn((12 * 64, 128), generator=gen, dtype=torch.float32, device=device) / 3
    )

    metadata = prepare_compressed_indexer_metadata(
        real_page_table=real_page_table,
        cache_seqlens_int32=seqlens,
        expected_num_q_heads=num_heads,
        build_schedule=True,
        schedule_num_sms=4,
    )
    with pytest.raises(NotImplementedError, match="production CUDA FP8 kernel contract"):
        compressed_index_decode_logits_fp8(
            q_fp8=q_fp8,
            weights=weights,
            index_k_cache=index_k_cache,
            metadata=metadata,
        )


def test_compressed_index_decode_rejects_sharded_selector_heads() -> None:
    device = torch.device("cpu")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(91_002)

    replicated_heads = resolve_replicated_num_q_heads(
        global_num_q_heads=64,
        tensor_parallel_size=2,
    )
    real_page_table = torch.tensor([[0]], dtype=torch.int32, device=device)
    seqlens = torch.tensor([1], dtype=torch.int32, device=device)
    metadata = prepare_compressed_indexer_metadata(
        real_page_table=real_page_table,
        cache_seqlens_int32=seqlens,
        expected_num_q_heads=replicated_heads,
        build_schedule=False,
    )
    q_fp8 = _rand_fp8_q((1, 32, 128), gen=gen, device=device)
    weights = torch.randn((1, 32), generator=gen, dtype=torch.float32, device=device)
    index_k_cache = pack_compressed_index_k_cache_reference(
        torch.randn((64, 128), generator=gen, dtype=torch.float32, device=device)
    )

    with pytest.raises(ValueError, match="expected indexer head count 64"):
        compressed_index_decode_logits_fp8(
            q_fp8=q_fp8,
            weights=weights,
            index_k_cache=index_k_cache,
            metadata=metadata,
        )


def test_compressed_index_metadata_rejects_clamp_to_one_lengths() -> None:
    real_page_table = torch.full((1, 2), -1, dtype=torch.int32)
    clamped_seqlens = torch.tensor([1], dtype=torch.int32)

    with pytest.raises(ValueError, match="raw unclamped compressed lengths"):
        prepare_compressed_indexer_metadata(
            real_page_table=real_page_table,
            cache_seqlens_int32=clamped_seqlens,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for graph capture")
def test_compressed_index_decode_logits_fp8_graph_workspace_matches_reference() -> None:
    device = torch.device("cuda")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(91_003)

    rows = 2
    local_heads = 32
    width_blocks = 1024
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    graph_real_page_table = torch.full(
        (rows, width_blocks),
        -1,
        dtype=torch.int32,
        device=device,
    )
    graph_seqlens = torch.empty((rows,), dtype=torch.int32, device=device)
    graph_schedule = torch.empty((num_sms + 1, 2), dtype=torch.int32, device=device)

    q_fp8 = _rand_fp8_q((rows, local_heads, 128), gen=gen, device=device)
    weights = torch.randn((rows, local_heads), generator=gen, dtype=torch.float32).to(device=device)
    index_k_cache = pack_compressed_index_k_cache_reference(
        torch.randn((1200 * 64, 128), generator=gen, dtype=torch.float32).to(device=device) / 3
    )
    workspace = B12XAttentionWorkspace.for_fixed_capacity(
        mode="decode",
        device=device,
        dtype=torch.bfloat16,
        kv_dtype=torch.float8_e4m3fn,
        num_q_heads=local_heads,
        indexer_num_q_heads=local_heads,
        head_dim=576,
        v_head_dim=512,
        topk=512,
        max_page_table_width=width_blocks,
        max_total_q=rows,
        max_batch=rows,
        max_paged_q_rows=rows,
        max_kv_rows=index_k_cache.shape[0] * 64,
        page_size=64,
        use_cuda_graph=True,
    )

    def prepare(page_starts: list[int], seqlens_list: list[int]):
        live_table = _make_real_page_table(
            page_starts=page_starts,
            seqlens=seqlens_list,
            width_blocks=width_blocks,
            device=device,
        )
        graph_real_page_table.copy_(live_table)
        graph_seqlens.copy_(torch.tensor(seqlens_list, dtype=torch.int32, device=device))
        return prepare_compressed_indexer_metadata(
            real_page_table=graph_real_page_table,
            cache_seqlens_int32=graph_seqlens,
            expected_num_q_heads=local_heads,
            schedule_out=graph_schedule,
        )

    clear_indexer_caches()
    metadata = prepare([2, 900], [2048, 2304])
    compressed_index_decode_logits_fp8(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        metadata=metadata,
        workspace=workspace,
    )
    torch.cuda.synchronize(device)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_out = compressed_index_decode_logits_fp8(
            q_fp8=q_fp8,
            weights=weights,
            index_k_cache=index_k_cache,
            metadata=metadata,
            workspace=workspace,
        )
    graph.replay()
    torch.cuda.synchronize(device)
    actual0 = captured_out.clone()
    expected0 = compressed_index_logits_reference(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        real_page_table=graph_real_page_table,
        query_row_to_batch=torch.arange(rows, dtype=torch.int32, device=device),
        seqlens_per_query=graph_seqlens,
    )
    torch.testing.assert_close(actual0, expected0, atol=1e-4, rtol=1e-4)

    prepare([4, 920], [65, 128])
    graph.replay()
    torch.cuda.synchronize(device)
    actual1 = captured_out.clone()
    expected1 = compressed_index_logits_reference(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        real_page_table=graph_real_page_table,
        query_row_to_batch=torch.arange(rows, dtype=torch.int32, device=device),
        seqlens_per_query=graph_seqlens,
    )
    torch.testing.assert_close(actual1, expected1, atol=1e-4, rtol=1e-4)
    assert torch.isneginf(actual1[:, 128:]).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for graph capture")
def test_compressed_index_decode_dense_topk_fp8_graph_matches_reference() -> None:
    device = torch.device("cuda")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(91_006)

    rows = 2
    num_heads = 64
    width_blocks = 16
    topk = 512
    graph_real_page_table = torch.full(
        (rows, width_blocks),
        -1,
        dtype=torch.int32,
        device=device,
    )
    graph_seqlens = torch.empty((rows,), dtype=torch.int32, device=device)
    q_fp8 = _rand_fp8_q((rows, num_heads, 128), gen=gen, device=device)
    weights = torch.randn((rows, num_heads), generator=gen, dtype=torch.float32).to(
        device=device
    )
    api_weights = weights.unsqueeze(-1)
    index_k_cache = pack_compressed_index_k_cache_reference(
        torch.randn((80 * 64, 128), generator=gen, dtype=torch.float32).to(device=device)
        / 3
    )
    workspace = B12XAttentionWorkspace.for_fixed_capacity(
        mode="decode",
        device=device,
        dtype=torch.bfloat16,
        kv_dtype=torch.float8_e4m3fn,
        num_q_heads=num_heads,
        indexer_num_q_heads=num_heads,
        head_dim=576,
        v_head_dim=512,
        topk=topk,
        max_page_table_width=width_blocks,
        max_total_q=rows,
        max_batch=rows,
        max_paged_q_rows=rows,
        max_kv_rows=0,
        page_size=64,
        use_cuda_graph=True,
        reserve_paged_indexer_logits=True,
        paged_indexer_logits_q_rows=rows,
        paged_indexer_logits_k_rows=width_blocks * 64,
        paged_indexer_tile_logits_k_rows=0,
    )
    actual = torch.empty((rows, topk), dtype=torch.int32, device=device)

    def prepare(page_starts: list[int], seqlens_list: list[int]):
        live_table = _make_real_page_table(
            page_starts=page_starts,
            seqlens=seqlens_list,
            width_blocks=width_blocks,
            device=device,
        )
        graph_real_page_table.copy_(live_table)
        graph_seqlens.copy_(torch.tensor(seqlens_list, dtype=torch.int32, device=device))
        return prepare_compressed_indexer_metadata(
            real_page_table=graph_real_page_table,
            cache_seqlens_int32=graph_seqlens,
            expected_num_q_heads=num_heads,
            build_schedule=False,
        )

    clear_indexer_caches()
    metadata = prepare([2, 40], [900, 960])
    compressed_index_decode_dense_topk_fp8(
        q_fp8=q_fp8,
        weights=api_weights,
        index_k_cache=index_k_cache,
        metadata=metadata,
        topk=topk,
        expected_num_q_heads=num_heads,
        workspace=workspace,
        out_indices=actual,
    )
    torch.cuda.synchronize(device)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compressed_index_decode_dense_topk_fp8(
            q_fp8=q_fp8,
            weights=api_weights,
            index_k_cache=index_k_cache,
            metadata=metadata,
            topk=topk,
            expected_num_q_heads=num_heads,
            workspace=workspace,
            out_indices=actual,
        )
    graph.replay()
    torch.cuda.synchronize(device)
    expected_raw0 = _expected_compressed_index_topk(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        real_page_table=graph_real_page_table,
        seqlens=graph_seqlens,
        topk=topk,
    )
    assert torch.equal(
        torch.sort(actual, dim=1).values,
        torch.sort(expected_raw0, dim=1).values,
    )

    prepare([4, 8], [640, 768])
    graph.replay()
    torch.cuda.synchronize(device)
    expected_raw1 = _expected_compressed_index_topk(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        real_page_table=graph_real_page_table,
        seqlens=graph_seqlens,
        topk=topk,
    )
    assert torch.equal(
        torch.sort(actual, dim=1).values,
        torch.sort(expected_raw1, dim=1).values,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for workspace allocation")
def test_compressed_index_decode_stage_keeps_contiguous_metadata_aliases() -> None:
    device = torch.device("cuda")
    rows = 1
    num_heads = 64
    width_blocks = 16
    workspace = B12XAttentionWorkspace.for_fixed_capacity(
        mode="decode",
        device=device,
        dtype=torch.bfloat16,
        kv_dtype=torch.float8_e4m3fn,
        num_q_heads=num_heads,
        indexer_num_q_heads=num_heads,
        head_dim=576,
        v_head_dim=512,
        topk=512,
        max_page_table_width=width_blocks,
        max_total_q=rows,
        max_batch=rows,
        max_paged_q_rows=rows,
        max_kv_rows=0,
        page_size=64,
        use_cuda_graph=True,
        reserve_paged_indexer_logits=True,
        paged_indexer_logits_q_rows=rows,
        paged_indexer_logits_k_rows=width_blocks * 64,
        paged_indexer_tile_logits_k_rows=0,
    )
    q_fp8 = torch.empty((rows, num_heads, 128), dtype=torch.float8_e4m3fn, device=device)
    weights = torch.empty((rows, num_heads), dtype=torch.float32, device=device)
    real_page_table = torch.empty((rows, width_blocks), dtype=torch.int32, device=device)
    seqlens = torch.empty((rows,), dtype=torch.int32, device=device)
    active_width = workspace.get_paged_indexer_active_width_cap()
    schedule = torch.empty((4, 2), dtype=torch.int32, device=device)

    staged = workspace.stage_indexer_paged_decode(
        q_fp8=q_fp8,
        weights=weights,
        real_page_table=real_page_table,
        seqlens_per_query=seqlens,
        active_width=active_width,
        schedule_metadata=schedule,
        width_tokens=width_blocks * 64,
        preinitialize_invalid_logits=False,
    )

    assert staged["real_page_table"].data_ptr() == real_page_table.data_ptr()
    assert staged["seqlens_per_query"].data_ptr() == seqlens.data_ptr()
    assert staged["active_width"].data_ptr() == active_width.data_ptr()
    assert staged["schedule_metadata"].data_ptr() == schedule.data_ptr()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for graph capture")
def test_compressed_index_decode_dense_topk_fp8_compacts_padded_page_table() -> None:
    device = torch.device("cuda")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(91_007)

    rows = 2
    num_heads = 64
    full_width_blocks = 4097
    workspace_width_blocks = 384
    topk = 512
    graph_real_page_table = torch.full(
        (rows, full_width_blocks),
        -1,
        dtype=torch.int32,
        device=device,
    )
    graph_seqlens = torch.empty((rows,), dtype=torch.int32, device=device)
    q_fp8 = _rand_fp8_q((rows, num_heads, 128), gen=gen, device=device)
    weights = torch.randn((rows, num_heads), generator=gen, dtype=torch.float32).to(
        device=device
    )
    api_weights = weights.unsqueeze(-1)
    index_k_cache = pack_compressed_index_k_cache_reference(
        torch.randn((96 * 64, 128), generator=gen, dtype=torch.float32).to(device=device)
        / 3
    )
    workspace = B12XAttentionWorkspace.for_fixed_capacity(
        mode="decode",
        device=device,
        dtype=torch.bfloat16,
        kv_dtype=torch.float8_e4m3fn,
        num_q_heads=num_heads,
        indexer_num_q_heads=num_heads,
        head_dim=576,
        v_head_dim=512,
        topk=topk,
        max_page_table_width=workspace_width_blocks,
        max_total_q=rows,
        max_batch=rows,
        max_paged_q_rows=rows,
        max_kv_rows=0,
        page_size=64,
        use_cuda_graph=True,
        reserve_paged_indexer_logits=True,
        paged_indexer_logits_q_rows=rows,
        paged_indexer_logits_k_rows=workspace_width_blocks * 64,
        paged_indexer_tile_logits_k_rows=0,
    )
    actual = torch.empty((rows, topk), dtype=torch.int32, device=device)

    def prepare(page_starts: list[int], seqlens_list: list[int]):
        live_table = _make_real_page_table(
            page_starts=page_starts,
            seqlens=seqlens_list,
            width_blocks=full_width_blocks,
            device=device,
        )
        graph_real_page_table.copy_(live_table)
        graph_seqlens.copy_(torch.tensor(seqlens_list, dtype=torch.int32, device=device))
        return prepare_compressed_indexer_metadata(
            real_page_table=graph_real_page_table,
            cache_seqlens_int32=graph_seqlens,
            expected_num_q_heads=num_heads,
            build_schedule=False,
        )

    clear_indexer_caches()
    metadata = prepare([2, 40], [900, 960])
    compressed_index_decode_dense_topk_fp8(
        q_fp8=q_fp8,
        weights=api_weights,
        index_k_cache=index_k_cache,
        metadata=metadata,
        topk=topk,
        expected_num_q_heads=num_heads,
        workspace=workspace,
        out_indices=actual,
    )
    torch.cuda.synchronize(device)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compressed_index_decode_dense_topk_fp8(
            q_fp8=q_fp8,
            weights=api_weights,
            index_k_cache=index_k_cache,
            metadata=metadata,
            topk=topk,
            expected_num_q_heads=num_heads,
            workspace=workspace,
            out_indices=actual,
        )
    graph.replay()
    torch.cuda.synchronize(device)
    expected_raw0 = _expected_compressed_index_topk(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        real_page_table=graph_real_page_table,
        seqlens=graph_seqlens,
        topk=topk,
    )
    assert torch.equal(
        torch.sort(actual, dim=1).values,
        torch.sort(expected_raw0, dim=1).values,
    )

    prepare([4, 8], [640, 768])
    graph.replay()
    torch.cuda.synchronize(device)
    expected_raw1 = _expected_compressed_index_topk(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        real_page_table=graph_real_page_table,
        seqlens=graph_seqlens,
        topk=topk,
    )
    assert torch.equal(
        torch.sort(actual, dim=1).values,
        torch.sort(expected_raw1, dim=1).values,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_compressed_index_decode_dense_topk_fp8_rekeys_compact_page_widths() -> None:
    device = torch.device("cuda")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(91_009)

    rows = 1
    num_heads = 64
    first_width_blocks = 8
    second_width_blocks = 16
    workspace_width_blocks = 64
    topk = 512
    q_fp8 = _rand_fp8_q((rows, num_heads, 128), gen=gen, device=device)
    weights = torch.randn((rows, num_heads), generator=gen, dtype=torch.float32).to(
        device=device
    )
    api_weights = weights.unsqueeze(-1)
    index_k_cache = pack_compressed_index_k_cache_reference(
        torch.randn((96 * 64, 128), generator=gen, dtype=torch.float32).to(device=device)
        / 3
    )
    workspace = B12XAttentionWorkspace.for_fixed_capacity(
        mode="decode",
        device=device,
        dtype=torch.bfloat16,
        kv_dtype=torch.float8_e4m3fn,
        num_q_heads=num_heads,
        indexer_num_q_heads=num_heads,
        head_dim=576,
        v_head_dim=512,
        topk=topk,
        max_page_table_width=workspace_width_blocks,
        max_total_q=rows,
        max_batch=rows,
        max_paged_q_rows=rows,
        max_kv_rows=0,
        page_size=64,
        reserve_paged_indexer_logits=True,
    )
    actual = torch.empty((rows, topk), dtype=torch.int32, device=device)

    def run(width_blocks: int, seqlen: int) -> torch.Tensor:
        page_table = _make_real_page_table(
            page_starts=[2],
            seqlens=[seqlen],
            width_blocks=width_blocks,
            device=device,
        )
        seqlens = torch.tensor([seqlen], dtype=torch.int32, device=device)
        metadata = prepare_compressed_indexer_metadata(
            real_page_table=page_table,
            cache_seqlens_int32=seqlens,
            expected_num_q_heads=num_heads,
            build_schedule=False,
        )
        return compressed_index_decode_dense_topk_fp8(
            q_fp8=q_fp8,
            weights=api_weights,
            index_k_cache=index_k_cache,
            metadata=metadata,
            topk=topk,
            expected_num_q_heads=num_heads,
            workspace=workspace,
            out_indices=actual,
        ).clone()

    clear_indexer_caches()
    first = run(first_width_blocks, 1)
    torch.cuda.synchronize(device)
    assert int(first[0, 0]) == 0

    second = run(second_width_blocks, 900)
    torch.cuda.synchronize(device)
    expected = _expected_compressed_index_topk(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        real_page_table=_make_real_page_table(
            page_starts=[2],
            seqlens=[900],
            width_blocks=second_width_blocks,
            device=device,
        ),
        seqlens=torch.tensor([900], dtype=torch.int32, device=device),
        topk=topk,
    )
    assert torch.equal(
        torch.sort(second, dim=1).values,
        torch.sort(expected, dim=1).values,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for graph capture")
def test_compressed_index_decode_supertile_topk_fp8_graph_matches_reference(
    monkeypatch,
) -> None:
    monkeypatch.setenv("B12X_COMPRESSED_INDEX_SUPERTILE_K", "512")

    device = torch.device("cuda")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(91_004)

    rows = 2
    num_heads = 64
    width_blocks = 16
    topk = 512
    graph_real_page_table = torch.full(
        (rows, width_blocks),
        -1,
        dtype=torch.int32,
        device=device,
    )
    graph_seqlens = torch.empty((rows,), dtype=torch.int32, device=device)
    q_fp8 = _rand_fp8_q((rows, num_heads, 128), gen=gen, device=device)
    weights = torch.randn((rows, num_heads), generator=gen, dtype=torch.float32).to(
        device=device
    )
    api_weights = weights.unsqueeze(-1)
    index_k_cache = pack_compressed_index_k_cache_reference(
        torch.randn((80 * 64, 128), generator=gen, dtype=torch.float32).to(device=device)
        / 3
    )
    workspace = B12XAttentionWorkspace.for_fixed_capacity(
        mode="decode",
        device=device,
        dtype=torch.bfloat16,
        kv_dtype=torch.float8_e4m3fn,
        num_q_heads=num_heads,
        indexer_num_q_heads=num_heads,
        head_dim=576,
        v_head_dim=512,
        topk=topk,
        max_page_table_width=width_blocks,
        max_total_q=rows,
        max_batch=rows,
        max_paged_q_rows=rows,
        max_kv_rows=0,
        page_size=64,
        use_cuda_graph=True,
        reserve_paged_indexer_logits=False,
        paged_indexer_tile_logits_k_rows=512,
    )
    actual = torch.empty((rows, topk), dtype=torch.int32, device=device)

    def prepare(page_starts: list[int], seqlens_list: list[int]):
        live_table = _make_real_page_table(
            page_starts=page_starts,
            seqlens=seqlens_list,
            width_blocks=width_blocks,
            device=device,
        )
        graph_real_page_table.copy_(live_table)
        graph_seqlens.copy_(torch.tensor(seqlens_list, dtype=torch.int32, device=device))
        return prepare_compressed_indexer_metadata(
            real_page_table=graph_real_page_table,
            cache_seqlens_int32=graph_seqlens,
            expected_num_q_heads=num_heads,
            build_schedule=False,
        )

    clear_indexer_caches()
    workspace.prewarm_paged_indexer_tiled_topk()
    workspace.prewarm_paged_indexer_tiled_scorer(
        index_k_cache=index_k_cache,
        width_tokens=512,
    )

    def fail_runtime_staging(**_kwargs):
        raise AssertionError("C4 supertile path must not stage workspace metadata at runtime")

    monkeypatch.setattr(workspace, "stage_indexer_paged_tiled_decode", fail_runtime_staging)

    metadata = prepare([2, 40], [900, 960])
    compressed_index_decode_supertile_topk_fp8(
        q_fp8=q_fp8,
        weights=api_weights,
        index_k_cache=index_k_cache,
        metadata=metadata,
        topk=topk,
        expected_num_q_heads=num_heads,
        workspace=workspace,
        out_indices=actual,
        supertile_k=512,
    )
    torch.cuda.synchronize(device)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compressed_index_decode_supertile_topk_fp8(
            q_fp8=q_fp8,
            weights=api_weights,
            index_k_cache=index_k_cache,
            metadata=metadata,
            topk=topk,
            expected_num_q_heads=num_heads,
            workspace=workspace,
            out_indices=actual,
            supertile_k=512,
        )
    graph.replay()
    torch.cuda.synchronize(device)
    expected_raw0 = _expected_compressed_index_topk(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        real_page_table=graph_real_page_table,
        seqlens=graph_seqlens,
        topk=topk,
    )
    assert torch.equal(
        torch.sort(actual, dim=1).values,
        torch.sort(expected_raw0, dim=1).values,
    )

    prepare([4, 8], [640, 768])
    graph.replay()
    torch.cuda.synchronize(device)
    expected_raw1 = _expected_compressed_index_topk(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        real_page_table=graph_real_page_table,
        seqlens=graph_seqlens,
        topk=topk,
    )
    assert torch.equal(
        torch.sort(actual, dim=1).values,
        torch.sort(expected_raw1, dim=1).values,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for graph capture")
def test_compressed_index_shared_supertile_prefill_graph_matches_reference(
    monkeypatch,
) -> None:
    monkeypatch.setenv("B12X_COMPRESSED_INDEX_SUPERTILE_K", "4096")

    device = torch.device("cuda")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(91_006)

    rows = 1024
    num_heads = 32
    width_blocks = 64
    topk = 512
    supertile_k = 4096
    graph_real_page_table = torch.full(
        (rows, width_blocks),
        -1,
        dtype=torch.int32,
        device=device,
    )
    graph_seqlens = torch.empty((rows,), dtype=torch.int32, device=device)
    q_fp8 = _rand_fp8_q((rows, num_heads, 128), gen=gen, device=device)
    weights = torch.randn((rows, num_heads), generator=gen, dtype=torch.float32).to(
        device=device
    )
    api_weights = weights.unsqueeze(-1)
    index_k_cache = pack_compressed_index_k_cache_reference(
        torch.randn((128 * 64, 128), generator=gen, dtype=torch.float32).to(device=device)
        / 3
    )
    workspace = B12XAttentionWorkspace.for_fixed_capacity(
        mode="decode",
        device=device,
        dtype=torch.bfloat16,
        kv_dtype=torch.float8_e4m3fn,
        num_q_heads=num_heads,
        indexer_num_q_heads=num_heads,
        head_dim=576,
        v_head_dim=512,
        topk=topk,
        max_page_table_width=width_blocks,
        max_total_q=1,
        max_batch=1,
        max_paged_q_rows=rows,
        max_kv_rows=0,
        indexer_max_k_rows=supertile_k,
        page_size=64,
        use_cuda_graph=True,
        reserve_paged_indexer_logits=False,
        paged_indexer_tile_logits_k_rows=supertile_k,
    )
    reference_workspace = B12XAttentionWorkspace.for_fixed_capacity(
        mode="decode",
        device=device,
        dtype=torch.bfloat16,
        kv_dtype=torch.float8_e4m3fn,
        num_q_heads=num_heads,
        indexer_num_q_heads=num_heads,
        head_dim=576,
        v_head_dim=512,
        topk=topk,
        max_page_table_width=width_blocks,
        max_total_q=rows,
        max_batch=rows,
        max_paged_q_rows=rows,
        max_kv_rows=0,
        page_size=64,
        use_cuda_graph=True,
        reserve_paged_indexer_logits=False,
        paged_indexer_tile_logits_k_rows=supertile_k,
    )
    actual = torch.empty((rows, topk), dtype=torch.int32, device=device)
    expected = torch.empty((rows, topk), dtype=torch.int32, device=device)

    def prepare(page_start: int, seq_len: int, *, shared_page_table: bool):
        base_table = _make_real_page_table(
            page_starts=[page_start],
            seqlens=[seq_len],
            width_blocks=width_blocks,
            device=device,
        )
        graph_real_page_table.copy_(base_table.expand(rows, -1))
        graph_seqlens.fill_(seq_len)
        return prepare_compressed_indexer_metadata(
            real_page_table=graph_real_page_table,
            cache_seqlens_int32=graph_seqlens,
            expected_num_q_heads=num_heads,
            build_schedule=False,
            shared_page_table=shared_page_table,
        )

    clear_indexer_caches()
    workspace.prewarm_paged_indexer_tiled_topk()
    workspace.prewarm_paged_indexer_tiled_scorer(
        index_k_cache=index_k_cache,
        width_tokens=supertile_k,
    )
    reference_workspace.prewarm_paged_indexer_tiled_topk()
    reference_workspace.prewarm_paged_indexer_tiled_scorer(
        index_k_cache=index_k_cache,
        width_tokens=supertile_k,
    )

    def fail_paged_scorer_staging(**_kwargs):
        raise AssertionError("shared C4 prefill must not use the row-wise paged scorer")

    monkeypatch.setattr(
        workspace,
        "stage_indexer_paged_tiled_decode",
        fail_paged_scorer_staging,
    )

    metadata = prepare(3, 4096, shared_page_table=True)
    reference_metadata = prepare(3, 4096, shared_page_table=False)
    compressed_index_decode_supertile_topk_fp8(
        q_fp8=q_fp8,
        weights=api_weights,
        index_k_cache=index_k_cache,
        metadata=reference_metadata,
        topk=topk,
        expected_num_q_heads=num_heads,
        workspace=reference_workspace,
        out_indices=expected,
        supertile_k=supertile_k,
    )
    compressed_index_decode_supertile_topk_fp8(
        q_fp8=q_fp8,
        weights=api_weights,
        index_k_cache=index_k_cache,
        metadata=metadata,
        topk=topk,
        expected_num_q_heads=num_heads,
        workspace=workspace,
        out_indices=actual,
        supertile_k=supertile_k,
    )
    torch.cuda.synchronize(device)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compressed_index_decode_supertile_topk_fp8(
            q_fp8=q_fp8,
            weights=api_weights,
            index_k_cache=index_k_cache,
            metadata=metadata,
            topk=topk,
            expected_num_q_heads=num_heads,
            workspace=workspace,
            out_indices=actual,
            supertile_k=supertile_k,
        )
    graph.replay()
    torch.cuda.synchronize(device)
    assert torch.equal(
        torch.sort(actual, dim=1).values,
        torch.sort(expected, dim=1).values,
    )

    metadata = prepare(24, 3584, shared_page_table=True)
    reference_metadata = prepare(24, 3584, shared_page_table=False)
    compressed_index_decode_supertile_topk_fp8(
        q_fp8=q_fp8,
        weights=api_weights,
        index_k_cache=index_k_cache,
        metadata=reference_metadata,
        topk=topk,
        expected_num_q_heads=num_heads,
        workspace=reference_workspace,
        out_indices=expected,
        supertile_k=supertile_k,
    )
    graph.replay()
    torch.cuda.synchronize(device)
    assert torch.equal(
        torch.sort(actual, dim=1).values,
        torch.sort(expected, dim=1).values,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for workspace allocation")
def test_compressed_index_supertile_workspace_sizes_candidate_chunks() -> None:
    device = torch.device("cuda")
    page_size = 256
    page_table_width = 1056
    supertile_k = 8192
    expected_chunks = (page_table_width * page_size + supertile_k - 1) // supertile_k

    arena = B12XAttentionArena.allocate(
        B12XAttentionArenaCaps(
            device=device,
            dtype=torch.bfloat16,
            kv_dtype=torch.float8_e4m3fn,
            num_q_heads=32,
            indexer_num_q_heads=64,
            head_dim=576,
            max_v_head_dim=512,
            topk=512,
            max_page_table_width=page_table_width,
            extend_max_total_q=16,
            extend_max_batch=4,
            extend_max_kv_rows=0,
            paged_max_q_rows=16,
            paged_max_batch=4,
            page_size=page_size,
            max_chunks_per_row=20,
            reserve_paged_indexer_logits=False,
            paged_indexer_tile_logits_k_rows=supertile_k,
        )
    )
    workspace = arena.make_workspace(
        B12XAttentionWorkspaceContract(
            mode="decode",
            max_total_q=16,
            max_batch=4,
            max_paged_q_rows=16,
            max_kv_rows=0,
            v_head_dim=512,
            indexer_num_q_heads=64,
            max_page_table_width=page_table_width,
            topk=512,
        ),
        use_cuda_graph=True,
    )

    candidate_values, candidate_indices = workspace.get_indexer_extend_candidate_buffers()
    assert candidate_values.shape[0] >= expected_chunks
    assert candidate_indices.shape[0] >= expected_chunks


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for workspace allocation")
def test_compressed_index_supertile_workspace_requires_prewarmed_launch_contract() -> None:
    device = torch.device("cuda")
    workspace = B12XAttentionWorkspace.for_fixed_capacity(
        mode="decode",
        device=device,
        dtype=torch.bfloat16,
        kv_dtype=torch.float8_e4m3fn,
        num_q_heads=64,
        indexer_num_q_heads=64,
        head_dim=576,
        v_head_dim=512,
        topk=512,
        max_page_table_width=16,
        max_total_q=2,
        max_batch=2,
        max_paged_q_rows=2,
        max_kv_rows=0,
        page_size=64,
        use_cuda_graph=True,
        reserve_paged_indexer_logits=False,
        paged_indexer_tile_logits_k_rows=512,
    )

    with pytest.raises(RuntimeError, match="not prewarmed"):
        workspace.require_paged_indexer_tiled_topk_plan(
            topk=512,
            block_q=32,
            block_k=512,
            num_k_tiles=1,
        )

    workspace.prewarm_paged_indexer_tiled_topk()
    with pytest.raises(RuntimeError, match="not prewarmed"):
        workspace.require_paged_indexer_tiled_scorer_plan(
            block_q=32,
            block_k=512,
            width_tokens=512,
            source_page_width=16,
        )
    index_k_cache = torch.empty((16, 64 * (128 + 4)), dtype=torch.uint8, device=device)
    workspace.prewarm_paged_indexer_tiled_scorer(
        index_k_cache=index_k_cache,
        width_tokens=512,
    )
    with pytest.raises(RuntimeError, match="does not match"):
        workspace.require_paged_indexer_tiled_scorer_plan(
            block_q=32,
            block_k=512,
            width_tokens=1024,
            source_page_width=16,
        )
    with pytest.raises(RuntimeError, match="does not match"):
        workspace.require_paged_indexer_tiled_topk_plan(
            topk=512,
            block_q=32,
            block_k=512,
            num_k_tiles=2,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for graph capture")
def test_compressed_index_decode_supertile_topk_fp8_graph_unaligned_single_chunk(
    monkeypatch,
) -> None:
    monkeypatch.setenv("B12X_COMPRESSED_INDEX_SUPERTILE_K", "1536")

    device = torch.device("cuda")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(91_005)

    rows = 2
    num_heads = 64
    width_blocks = 17
    supertile_blocks = 24
    topk = 512
    graph_real_page_table = torch.full(
        (rows, width_blocks),
        -1,
        dtype=torch.int32,
        device=device,
    )
    graph_seqlens = torch.empty((rows,), dtype=torch.int32, device=device)
    q_fp8 = _rand_fp8_q((rows, num_heads, 128), gen=gen, device=device)
    weights = torch.randn((rows, num_heads), generator=gen, dtype=torch.float32).to(
        device=device
    )
    api_weights = weights.unsqueeze(-1)
    index_k_cache = pack_compressed_index_k_cache_reference(
        torch.randn((96 * 64, 128), generator=gen, dtype=torch.float32).to(device=device)
        / 3
    )
    workspace = B12XAttentionWorkspace.for_fixed_capacity(
        mode="decode",
        device=device,
        dtype=torch.bfloat16,
        kv_dtype=torch.float8_e4m3fn,
        num_q_heads=num_heads,
        indexer_num_q_heads=num_heads,
        head_dim=576,
        v_head_dim=512,
        topk=topk,
        max_page_table_width=supertile_blocks,
        max_total_q=rows,
        max_batch=rows,
        max_paged_q_rows=rows,
        max_kv_rows=0,
        page_size=64,
        use_cuda_graph=True,
        reserve_paged_indexer_logits=False,
        paged_indexer_tile_logits_k_rows=1536,
    )
    actual = torch.empty((rows, topk), dtype=torch.int32, device=device)

    def prepare(page_starts: list[int], seqlens_list: list[int]):
        live_table = _make_real_page_table(
            page_starts=page_starts,
            seqlens=seqlens_list,
            width_blocks=width_blocks,
            device=device,
        )
        graph_real_page_table.copy_(live_table)
        graph_seqlens.copy_(torch.tensor(seqlens_list, dtype=torch.int32, device=device))
        return prepare_compressed_indexer_metadata(
            real_page_table=graph_real_page_table,
            cache_seqlens_int32=graph_seqlens,
            expected_num_q_heads=num_heads,
            build_schedule=False,
        )

    clear_indexer_caches()
    workspace.prewarm_paged_indexer_tiled_topk()
    workspace.prewarm_paged_indexer_tiled_scorer(
        index_k_cache=index_k_cache,
        width_tokens=1536,
    )
    metadata = prepare([2, 48], [960, 1024])
    compressed_index_decode_supertile_topk_fp8(
        q_fp8=q_fp8,
        weights=api_weights,
        index_k_cache=index_k_cache,
        metadata=metadata,
        topk=topk,
        expected_num_q_heads=num_heads,
        workspace=workspace,
        out_indices=actual,
        supertile_k=1536,
    )
    torch.cuda.synchronize(device)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compressed_index_decode_supertile_topk_fp8(
            q_fp8=q_fp8,
            weights=api_weights,
            index_k_cache=index_k_cache,
            metadata=metadata,
            topk=topk,
            expected_num_q_heads=num_heads,
            workspace=workspace,
            out_indices=actual,
            supertile_k=1536,
        )
    graph.replay()
    torch.cuda.synchronize(device)
    expected_raw0 = _expected_compressed_index_topk(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        real_page_table=graph_real_page_table,
        seqlens=graph_seqlens,
        topk=topk,
    )
    assert torch.equal(
        torch.sort(actual, dim=1).values,
        torch.sort(expected_raw0, dim=1).values,
    )

    prepare([4, 12], [640, 704])
    graph.replay()
    torch.cuda.synchronize(device)
    expected_raw1 = _expected_compressed_index_topk(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        real_page_table=graph_real_page_table,
        seqlens=graph_seqlens,
        topk=topk,
    )
    assert torch.equal(
        torch.sort(actual, dim=1).values,
        torch.sort(expected_raw1, dim=1).values,
    )
