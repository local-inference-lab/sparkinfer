from __future__ import annotations

import pytest
import torch

from b12x.attention.indexer import (
    clear_indexer_caches,
    index_topk_fp8,
    pack_paged_index_k_cache_reference,
    paged_index_logits_reference,
    prepare_paged_indexer_metadata,
    resolve_replicated_num_q_heads,
)
from b12x.attention.indexer.scratch import (
    B12XIndexerPagedScratchCaps,
    plan_indexer_paged_scratch,
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


def _expected_paged_index_topk(
    *,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    index_k_cache: torch.Tensor,
    real_page_table: torch.Tensor,
    seqlens: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    logits = paged_index_logits_reference(
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


def _bind_paged_indexer(
    *,
    device: torch.device,
    num_heads: int,
    rows: int,
    width_blocks: int,
    topk: int,
    real_page_table: torch.Tensor,
    seqlens: torch.Tensor,
    supertile_k: int = 512,
    active_width: torch.Tensor | None = None,
    schedule_metadata: torch.Tensor | None = None,
    shared_page_table: bool = False,
    route: str = "paged_tiled",
):
    plan = plan_indexer_paged_scratch(
        B12XIndexerPagedScratchCaps(
            device=device,
            num_q_heads=num_heads,
            max_q_rows=rows,
            max_page_table_width=width_blocks,
            topk=topk,
            page_size=64,
            reserve_paged_logits=False,
            paged_tile_logits_k_rows=supertile_k,
            mode="prefill" if shared_page_table else "decode",
            shared_page_table=shared_page_table,
            route=route,
        )
    )
    scratch = [
        torch.empty(shape, dtype=dtype, device=device)
        for shape, dtype in plan.shapes_and_dtypes()
    ]
    return plan.bind(
        scratch=scratch,
        real_page_table=real_page_table,
        cache_seqlens_int32=seqlens,
        active_width=active_width,
        schedule_metadata=schedule_metadata,
        expected_num_q_heads=num_heads,
        shared_page_table=shared_page_table,
    )


def test_resolve_replicated_num_q_heads_for_tensor_parallel() -> None:
    assert resolve_replicated_num_q_heads(global_num_q_heads=64, tensor_parallel_size=2) == 64
    assert resolve_replicated_num_q_heads(global_num_q_heads=64, tensor_parallel_size=1) == 64
    with pytest.raises(ValueError, match="must be positive"):
        resolve_replicated_num_q_heads(global_num_q_heads=0, tensor_parallel_size=2)
    with pytest.raises(ValueError, match="tensor_parallel_size must be positive"):
        resolve_replicated_num_q_heads(global_num_q_heads=64, tensor_parallel_size=0)


def test_index_topk_fp8_hard_fails_on_cpu() -> None:
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
    index_k_cache = pack_paged_index_k_cache_reference(
        torch.randn((12 * 64, 128), generator=gen, dtype=torch.float32, device=device) / 3
    )

    schedule = torch.empty((5, 2), dtype=torch.int32, device=device)
    binding = _bind_paged_indexer(
        device=device,
        num_heads=num_heads,
        rows=rows,
        width_blocks=width_blocks,
        topk=512,
        real_page_table=real_page_table,
        seqlens=seqlens,
        schedule_metadata=schedule,
    )
    with pytest.raises(NotImplementedError, match="requires CUDA"):
        index_topk_fp8(
            q_fp8=q_fp8,
            weights=weights,
            index_k_cache=index_k_cache,
            topk=512,
            binding=binding,
        )


def test_paged_index_decode_rejects_sharded_selector_heads() -> None:
    device = torch.device("cpu")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(91_002)

    replicated_heads = resolve_replicated_num_q_heads(
        global_num_q_heads=64,
        tensor_parallel_size=2,
    )
    real_page_table = torch.tensor([[0]], dtype=torch.int32, device=device)
    seqlens = torch.tensor([1], dtype=torch.int32, device=device)
    binding = _bind_paged_indexer(
        device=device,
        num_heads=replicated_heads,
        rows=1,
        width_blocks=1,
        topk=512,
        real_page_table=real_page_table,
        seqlens=seqlens,
    )
    q_fp8 = _rand_fp8_q((1, 32, 128), gen=gen, device=device)
    weights = torch.randn((1, 32), generator=gen, dtype=torch.float32, device=device)
    index_k_cache = pack_paged_index_k_cache_reference(
        torch.randn((64, 128), generator=gen, dtype=torch.float32, device=device)
    )

    with pytest.raises(ValueError, match="expected indexer head count 64"):
        index_topk_fp8(
            q_fp8=q_fp8,
            weights=weights,
            index_k_cache=index_k_cache,
            topk=512,
            binding=binding,
        )


def test_paged_index_metadata_rejects_clamp_to_one_lengths() -> None:
    real_page_table = torch.full((1, 2), -1, dtype=torch.int32)
    clamped_seqlens = torch.tensor([1], dtype=torch.int32)

    with pytest.raises(ValueError, match="raw unclamped paged-index lengths"):
        prepare_paged_indexer_metadata(
            real_page_table=real_page_table,
            cache_seqlens_int32=clamped_seqlens,
        )


def test_paged_index_plan_binding_keeps_metadata_aliases() -> None:
    device = torch.device("cpu")
    rows = 1
    num_heads = 64
    width_blocks = 16
    q_fp8 = torch.empty((rows, num_heads, 128), dtype=torch.float8_e4m3fn, device=device)
    weights = torch.empty((rows, num_heads), dtype=torch.float32, device=device)
    del q_fp8, weights
    real_page_table = torch.empty((rows, width_blocks), dtype=torch.int32, device=device)
    seqlens = torch.empty((rows,), dtype=torch.int32, device=device)
    active_width = torch.empty((1,), dtype=torch.int32, device=device)
    schedule = torch.empty((4, 2), dtype=torch.int32, device=device)

    binding = _bind_paged_indexer(
        device=device,
        num_heads=num_heads,
        rows=rows,
        width_blocks=width_blocks,
        topk=512,
        real_page_table=real_page_table,
        seqlens=seqlens,
        active_width=active_width,
        schedule_metadata=schedule,
    )

    assert binding.real_page_table.data_ptr() == real_page_table.data_ptr()
    assert binding.cache_seqlens_int32.data_ptr() == seqlens.data_ptr()
    assert binding.active_width.data_ptr() == active_width.data_ptr()
    assert binding.schedule_metadata.data_ptr() == schedule.data_ptr()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for graph capture")
def test_index_topk_fp8_graph_matches_reference(
    monkeypatch,
) -> None:
    monkeypatch.setenv("B12X_PAGED_INDEX_SUPERTILE_K", "512")

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
    index_k_cache = pack_paged_index_k_cache_reference(
        torch.randn((80 * 64, 128), generator=gen, dtype=torch.float32).to(device=device)
        / 3
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
        return prepare_paged_indexer_metadata(
            real_page_table=graph_real_page_table,
            cache_seqlens_int32=graph_seqlens,
            expected_num_q_heads=num_heads,
            build_schedule=False,
        )

    binding = _bind_paged_indexer(
        device=device,
        num_heads=num_heads,
        rows=rows,
        width_blocks=width_blocks,
        topk=topk,
        real_page_table=graph_real_page_table,
        seqlens=graph_seqlens,
        supertile_k=512,
    )

    clear_indexer_caches()
    metadata = prepare([2, 40], [900, 960])
    assert metadata.real_page_table.data_ptr() == binding.real_page_table.data_ptr()
    index_topk_fp8(
        q_fp8=q_fp8,
        weights=api_weights,
        index_k_cache=index_k_cache,
        topk=topk,
        expected_num_q_heads=num_heads,
        binding=binding,
        out_indices=actual,
        supertile_k=512,
    )
    torch.cuda.synchronize(device)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        index_topk_fp8(
            q_fp8=q_fp8,
            weights=api_weights,
            index_k_cache=index_k_cache,
            topk=topk,
            expected_num_q_heads=num_heads,
            binding=binding,
            out_indices=actual,
            supertile_k=512,
        )
    graph.replay()
    torch.cuda.synchronize(device)
    expected_raw0 = _expected_paged_index_topk(
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
    expected_raw1 = _expected_paged_index_topk(
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
def test_paged_index_shared_supertile_prefill_graph_matches_reference(
    monkeypatch,
) -> None:
    monkeypatch.setenv("B12X_PAGED_INDEX_SUPERTILE_K", "4096")

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
    graph_shared_page_table = torch.full(
        (1, width_blocks),
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
    index_k_cache = pack_paged_index_k_cache_reference(
        torch.randn((128 * 64, 128), generator=gen, dtype=torch.float32).to(device=device)
        / 3
    )
    padded_index_k_cache = torch.empty(
        (int(index_k_cache.shape[0]), int(index_k_cache.shape[1]) + 16),
        dtype=index_k_cache.dtype,
        device=device,
    )
    padded_index_k_cache[:, : int(index_k_cache.shape[1])].copy_(index_k_cache)
    index_k_cache = padded_index_k_cache.as_strided(
        tuple(index_k_cache.shape),
        (int(padded_index_k_cache.stride(0)), 1),
    )
    assert not index_k_cache.is_contiguous()
    actual = torch.empty((rows, topk), dtype=torch.int32, device=device)
    expected = torch.empty((rows, topk), dtype=torch.int32, device=device)
    shared_table_binding = graph_shared_page_table.expand(rows, -1)
    binding = _bind_paged_indexer(
        device=device,
        num_heads=num_heads,
        rows=rows,
        width_blocks=width_blocks,
        topk=topk,
        real_page_table=shared_table_binding,
        seqlens=graph_seqlens,
        supertile_k=supertile_k,
        shared_page_table=True,
    )
    reference_binding = _bind_paged_indexer(
        device=device,
        num_heads=num_heads,
        rows=rows,
        width_blocks=width_blocks,
        topk=topk,
        real_page_table=graph_real_page_table,
        seqlens=graph_seqlens,
        supertile_k=supertile_k,
        shared_page_table=False,
    )

    def prepare(page_start: int, seq_len: int, *, shared_page_table: bool):
        base_table = _make_real_page_table(
            page_starts=[page_start],
            seqlens=[seq_len],
            width_blocks=width_blocks,
            device=device,
        )
        if shared_page_table:
            graph_shared_page_table.copy_(base_table)
            real_page_table = shared_table_binding
        else:
            graph_real_page_table.copy_(base_table.expand(rows, -1))
            real_page_table = graph_real_page_table
        graph_seqlens.fill_(seq_len)
        return prepare_paged_indexer_metadata(
            real_page_table=real_page_table,
            cache_seqlens_int32=graph_seqlens,
            expected_num_q_heads=num_heads,
            build_schedule=False,
            shared_page_table=shared_page_table,
        )

    clear_indexer_caches()
    metadata = prepare(3, 4096, shared_page_table=True)
    reference_metadata = prepare(3, 4096, shared_page_table=False)
    assert metadata.real_page_table.data_ptr() == binding.real_page_table.data_ptr()
    assert reference_metadata.real_page_table.data_ptr() == reference_binding.real_page_table.data_ptr()
    index_topk_fp8(
        q_fp8=q_fp8,
        weights=api_weights,
        index_k_cache=index_k_cache,
        topk=topk,
        expected_num_q_heads=num_heads,
        binding=reference_binding,
        out_indices=expected,
        supertile_k=supertile_k,
    )
    index_topk_fp8(
        q_fp8=q_fp8,
        weights=api_weights,
        index_k_cache=index_k_cache,
        topk=topk,
        expected_num_q_heads=num_heads,
        binding=binding,
        out_indices=actual,
        supertile_k=supertile_k,
    )
    torch.cuda.synchronize(device)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        index_topk_fp8(
            q_fp8=q_fp8,
            weights=api_weights,
            index_k_cache=index_k_cache,
            topk=topk,
            expected_num_q_heads=num_heads,
            binding=binding,
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
    index_topk_fp8(
        q_fp8=q_fp8,
        weights=api_weights,
        index_k_cache=index_k_cache,
        topk=topk,
        expected_num_q_heads=num_heads,
        binding=reference_binding,
        out_indices=expected,
        supertile_k=supertile_k,
    )
    graph.replay()
    torch.cuda.synchronize(device)
    assert torch.equal(
        torch.sort(actual, dim=1).values,
        torch.sort(expected, dim=1).values,
    )


def test_paged_index_supertile_scratch_sizes_candidate_carry_buffer() -> None:
    device = torch.device("cpu")
    page_size = 64
    page_table_width = 1056
    supertile_k = 8192
    # The streaming fold replaced the old map+merge candidate slab (one half per
    # supertile chunk) with a fixed two-half carry double-buffer (read prev / write
    # next), so the candidate buffer no longer scales with the chunk count.
    chunk_count = (page_table_width * page_size + supertile_k - 1) // supertile_k
    assert chunk_count > 2  # this config is genuinely multi-chunk

    plan = plan_indexer_paged_scratch(
        B12XIndexerPagedScratchCaps(
            device=device,
            num_q_heads=64,
            max_q_rows=16,
            max_page_table_width=page_table_width,
            topk=512,
            page_size=page_size,
            reserve_paged_logits=False,
            paged_tile_logits_k_rows=supertile_k,
            route="paged_tiled",
        )
    )
    scratch = [
        torch.empty(shape, dtype=dtype, device=device)
        for shape, dtype in plan.shapes_and_dtypes()
    ]
    real_page_table = torch.empty((16, page_table_width), dtype=torch.int32, device=device)
    seqlens = torch.empty((16,), dtype=torch.int32, device=device)
    binding = plan.bind(
        scratch=scratch,
        real_page_table=real_page_table,
        cache_seqlens_int32=seqlens,
        expected_num_q_heads=64,
    )
    candidate_values, candidate_indices = (
        binding.scratch.get_indexer_contiguous_candidate_buffers()
    )
    # Fold carry double-buffer: exactly two halves, decoupled from chunk_count.
    assert candidate_values.shape[0] == 2
    assert candidate_indices.shape[0] == 2


def test_paged_index_supertile_plan_records_launch_contract() -> None:
    device = torch.device("cpu")
    plan = plan_indexer_paged_scratch(
        B12XIndexerPagedScratchCaps(
            device=device,
            num_q_heads=64,
            max_q_rows=2,
            max_page_table_width=16,
            topk=512,
            page_size=64,
            reserve_paged_logits=False,
            paged_tile_logits_k_rows=512,
            route="paged_tiled",
        )
    )

    assert plan.layout.route == "paged_tiled"
    assert plan.layout.supertile_tokens == 512
    assert plan.layout.max_chunks == 2
    assert plan.layout.tile_logits_elements == 32 * 512


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for graph capture")
def test_index_topk_fp8_graph_unaligned_single_chunk(
    monkeypatch,
) -> None:
    monkeypatch.setenv("B12X_PAGED_INDEX_SUPERTILE_K", "1536")

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
    index_k_cache = pack_paged_index_k_cache_reference(
        torch.randn((96 * 64, 128), generator=gen, dtype=torch.float32).to(device=device)
        / 3
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
        return prepare_paged_indexer_metadata(
            real_page_table=graph_real_page_table,
            cache_seqlens_int32=graph_seqlens,
            expected_num_q_heads=num_heads,
            build_schedule=False,
        )

    binding = _bind_paged_indexer(
        device=device,
        num_heads=num_heads,
        rows=rows,
        width_blocks=supertile_blocks,
        topk=topk,
        real_page_table=graph_real_page_table,
        seqlens=graph_seqlens,
        supertile_k=1536,
    )
    clear_indexer_caches()
    metadata = prepare([2, 48], [960, 1024])
    assert metadata.real_page_table.data_ptr() == binding.real_page_table.data_ptr()
    index_topk_fp8(
        q_fp8=q_fp8,
        weights=api_weights,
        index_k_cache=index_k_cache,
        topk=topk,
        expected_num_q_heads=num_heads,
        binding=binding,
        out_indices=actual,
        supertile_k=1536,
    )
    torch.cuda.synchronize(device)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        index_topk_fp8(
            q_fp8=q_fp8,
            weights=api_weights,
            index_k_cache=index_k_cache,
            topk=topk,
            expected_num_q_heads=num_heads,
            binding=binding,
            out_indices=actual,
            supertile_k=1536,
        )
    graph.replay()
    torch.cuda.synchronize(device)
    expected_raw0 = _expected_paged_index_topk(
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
    expected_raw1 = _expected_paged_index_topk(
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
