from __future__ import annotations

import pytest
import torch

from b12x.integration import (
    B12XAttentionWorkspace,
    clear_nsa_indexer_caches,
    pack_paged_mqa_index_k_cache_reference,
    paged_mqa_index_decode_logits_fp8,
    paged_mqa_index_logits_reference,
    prepare_paged_mqa_indexer_metadata,
    resolve_local_num_q_heads,
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


def test_resolve_local_num_q_heads_for_tensor_parallel() -> None:
    assert resolve_local_num_q_heads(global_num_q_heads=64, tensor_parallel_size=2) == 32
    assert resolve_local_num_q_heads(global_num_q_heads=64, tensor_parallel_size=1) == 64
    with pytest.raises(ValueError, match="not divisible"):
        resolve_local_num_q_heads(global_num_q_heads=65, tensor_parallel_size=2)


def test_paged_mqa_index_decode_logits_fp8_matches_reference_cpu() -> None:
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
    index_k_cache = pack_paged_mqa_index_k_cache_reference(
        torch.randn((12 * 64, 128), generator=gen, dtype=torch.float32, device=device) / 3
    )

    metadata = prepare_paged_mqa_indexer_metadata(
        real_page_table=real_page_table,
        cache_seqlens_int32=seqlens,
        expected_num_q_heads=num_heads,
        build_schedule=True,
        schedule_num_sms=4,
    )
    actual = paged_mqa_index_decode_logits_fp8(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        metadata=metadata,
    )
    expected = paged_mqa_index_logits_reference(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        real_page_table=real_page_table,
        query_row_to_batch=torch.arange(rows, dtype=torch.int32, device=device),
        seqlens_per_query=seqlens,
    )

    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


def test_paged_mqa_index_decode_rejects_global_head_padding() -> None:
    device = torch.device("cpu")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(91_002)

    local_heads = resolve_local_num_q_heads(global_num_q_heads=64, tensor_parallel_size=2)
    real_page_table = torch.tensor([[0]], dtype=torch.int32, device=device)
    seqlens = torch.tensor([1], dtype=torch.int32, device=device)
    metadata = prepare_paged_mqa_indexer_metadata(
        real_page_table=real_page_table,
        cache_seqlens_int32=seqlens,
        expected_num_q_heads=local_heads,
        build_schedule=False,
    )
    q_fp8 = _rand_fp8_q((1, 64, 128), gen=gen, device=device)
    weights = torch.randn((1, 64), generator=gen, dtype=torch.float32, device=device)
    index_k_cache = pack_paged_mqa_index_k_cache_reference(
        torch.randn((64, 128), generator=gen, dtype=torch.float32, device=device)
    )

    with pytest.raises(ValueError, match="TP-local head count 32"):
        paged_mqa_index_decode_logits_fp8(
            q_fp8=q_fp8,
            weights=weights,
            index_k_cache=index_k_cache,
            metadata=metadata,
        )


def test_paged_mqa_index_metadata_rejects_clamp_to_one_lengths() -> None:
    real_page_table = torch.full((1, 2), -1, dtype=torch.int32)
    clamped_seqlens = torch.tensor([1], dtype=torch.int32)

    with pytest.raises(ValueError, match="raw unclamped compressed lengths"):
        prepare_paged_mqa_indexer_metadata(
            real_page_table=real_page_table,
            cache_seqlens_int32=clamped_seqlens,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for graph capture")
def test_paged_mqa_index_decode_logits_fp8_graph_workspace_matches_reference() -> None:
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
    index_k_cache = pack_paged_mqa_index_k_cache_reference(
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
        return prepare_paged_mqa_indexer_metadata(
            real_page_table=graph_real_page_table,
            cache_seqlens_int32=graph_seqlens,
            expected_num_q_heads=local_heads,
            schedule_out=graph_schedule,
        )

    clear_nsa_indexer_caches()
    metadata = prepare([2, 900], [2048, 2304])
    paged_mqa_index_decode_logits_fp8(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        metadata=metadata,
        workspace=workspace,
    )
    torch.cuda.synchronize(device)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_out = paged_mqa_index_decode_logits_fp8(
            q_fp8=q_fp8,
            weights=weights,
            index_k_cache=index_k_cache,
            metadata=metadata,
            workspace=workspace,
        )
    graph.replay()
    torch.cuda.synchronize(device)
    actual0 = captured_out.clone()
    expected0 = paged_mqa_index_logits_reference(
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
    expected1 = paged_mqa_index_logits_reference(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        real_page_table=graph_real_page_table,
        query_row_to_batch=torch.arange(rows, dtype=torch.int32, device=device),
        seqlens_per_query=graph_seqlens,
    )
    torch.testing.assert_close(actual1, expected1, atol=1e-4, rtol=1e-4)
    assert torch.isneginf(actual1[:, 128:]).all()
