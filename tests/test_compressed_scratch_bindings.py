from __future__ import annotations

import torch

import b12x.attention.indexer.api as indexer_impl
import b12x.attention.mla.api as sparse_mla_impl
import b12x.attention.mla.compressed_api as compressed_mla_impl
import b12x.integration.compressed_indexer as compressed_indexer_impl
from b12x.attention.mla.compressed_reference import (
    COMPRESSED_MLA_DSV4_PAGE_SIZE,
    compressed_mla_page_nbytes,
)
from b12x.attention.workspace import B12XAttentionWorkspace
from b12x.integration import (
    B12XCompressedIndexerBinding,
    B12XCompressedIndexerScratchCaps,
    B12XCompressedMLABinding,
    B12XCompressedMLAScratch,
    B12XCompressedMLAScratchCaps,
    B12XIndexerExtendBinding,
    B12XIndexerExtendScratchCaps,
    B12XIndexerPagedBinding,
    B12XIndexerPagedScratchCaps,
    B12XSparseMLABinding,
    B12XSparseMLAScratchCaps,
    plan_compressed_indexer_scratch,
    plan_compressed_mla_scratch,
    plan_indexer_extend_scratch,
    plan_indexer_paged_scratch,
    plan_sparse_mla_scratch,
)


def _workspace(
    *,
    num_q_heads: int = 2,
    indexer_num_q_heads: int = 2,
    max_total_q: int = 4,
    max_paged_q_rows: int = 4,
    topk: int = 8,
    max_page_table_width: int = 8,
) -> B12XAttentionWorkspace:
    return B12XAttentionWorkspace(
        mode="decode",
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        kv_dtype=torch.uint8,
        num_q_heads=num_q_heads,
        indexer_num_q_heads=indexer_num_q_heads,
        head_dim=512,
        v_head_dim=512,
        topk=topk,
        indexer_topk=topk,
        max_page_table_width=max_page_table_width,
        max_total_q=max_total_q,
        max_batch=max_total_q,
        max_paged_q_rows=max_paged_q_rows,
        max_kv_rows=0,
        fixed_capacity=True,
        max_chunks_per_row=4,
    )


def test_compressed_mla_scratch_plan_exposes_one_opaque_scratch_spec() -> None:
    plan = plan_compressed_mla_scratch(
        B12XCompressedMLAScratchCaps(
            device="cpu",
            num_q_heads=2,
            max_q_rows=4,
            max_width=8,
            max_page_table_width=16,
        )
    )

    specs = plan.scratch_specs()
    assert len(specs) == 1
    assert specs[0].name == "compressed_mla.scratch"
    assert specs[0].dtype == torch.uint8
    assert specs[0].shape == plan.shapes_and_dtypes()[0][0]
    assert specs[0].nbytes == specs[0].shape[0]
    assert plan.layout.nbytes == specs[0].nbytes


def test_compressed_mla_scratch_binding_uses_component_scratch() -> None:
    plan = plan_compressed_mla_scratch(
        B12XCompressedMLAScratchCaps(
            device="cpu",
            num_q_heads=2,
            max_q_rows=4,
            max_width=8,
            max_page_table_width=16,
        )
    )
    (spec,) = plan.scratch_specs()
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    q = torch.empty((4, 2, 512), dtype=torch.bfloat16)
    swa_indices = torch.empty((4, 8), dtype=torch.int32)
    swa_lengths = torch.empty((4,), dtype=torch.int32)

    binding = plan.bind(
        scratch=scratch,
        q=q,
        swa_indices=swa_indices,
        swa_lengths=swa_lengths,
    )

    assert isinstance(binding.scratch, B12XCompressedMLAScratch)
    assert binding.scratch.shared_scratch.data_ptr() == scratch.data_ptr()
    assert binding.scratch.tmp_output is not None
    assert binding.scratch.tmp_lse is not None
    assert binding.scratch.output_buffer is not None
    assert binding.scratch.kv_chunk_size_ptr is not None
    assert binding.scratch.num_chunks_ptr is not None
    assert not hasattr(binding.scratch, "indexer_k_tma_desc_ptrs")


def test_compressed_indexer_scratch_plan_exposes_one_opaque_arena_spec() -> None:
    plan = plan_compressed_indexer_scratch(
        B12XCompressedIndexerScratchCaps(
            device="cpu",
            num_q_heads=2,
            max_q_rows=4,
            max_page_table_width=16,
            topk=8,
            reserve_paged_logits=False,
            paged_tile_logits_k_rows=512,
        )
    )

    specs = plan.scratch_specs()
    assert len(specs) == 1
    assert specs[0].name == "compressed_indexer.arena"
    assert specs[0].dtype == torch.uint8
    assert specs[0].shape == plan.shapes_and_dtypes()[0][0]
    assert specs[0].nbytes == specs[0].shape[0]
    assert plan.arena_caps.reserve_paged_indexer_logits is False


def test_indexer_paged_scratch_plan_exposes_one_opaque_arena_spec() -> None:
    plan = plan_indexer_paged_scratch(
        B12XIndexerPagedScratchCaps(
            device="cpu",
            num_q_heads=2,
            max_q_rows=4,
            max_page_table_width=16,
            reserve_paged_logits=False,
            paged_tile_logits_k_rows=512,
        )
    )

    specs = plan.scratch_specs()
    assert len(specs) == 1
    assert specs[0].name == "indexer_paged.arena"
    assert specs[0].dtype == torch.uint8
    assert specs[0].shape == plan.shapes_and_dtypes()[0][0]
    assert specs[0].nbytes == specs[0].shape[0]
    assert plan.arena_caps.reserve_paged_indexer_logits is False


def test_indexer_extend_scratch_plan_exposes_one_opaque_arena_spec() -> None:
    plan = plan_indexer_extend_scratch(
        B12XIndexerExtendScratchCaps(
            device="cpu",
            num_q_heads=2,
            max_q_rows=4,
            max_k_rows=1024,
            topk=8,
            reserve_extend_logits=False,
            extend_tile_logits_k_rows=512,
        )
    )

    specs = plan.scratch_specs()
    assert len(specs) == 1
    assert specs[0].name == "indexer_extend.arena"
    assert specs[0].dtype == torch.uint8
    assert specs[0].shape == plan.shapes_and_dtypes()[0][0]
    assert specs[0].nbytes == specs[0].shape[0]
    assert plan.contract.mode == "extend"
    assert plan.arena_caps.reserve_extend_indexer_logits is False
    assert plan.arena_caps.indexer_max_k_rows == 1024


def test_sparse_mla_scratch_plan_exposes_one_opaque_arena_spec() -> None:
    plan = plan_sparse_mla_scratch(
        B12XSparseMLAScratchCaps(
            device="cpu",
            num_q_heads=2,
            max_q_rows=4,
            max_width=8,
            max_page_table_width=16,
            head_dim=512,
            v_head_dim=512,
            mode="extend",
        )
    )

    specs = plan.scratch_specs()
    assert len(specs) == 1
    assert specs[0].name == "sparse_mla.arena"
    assert specs[0].dtype == torch.uint8
    assert specs[0].shape == plan.shapes_and_dtypes()[0][0]
    assert specs[0].nbytes == specs[0].shape[0]
    assert plan.contract.mode == "extend"


def test_workspace_bind_compressed_mla_returns_common_binding_type() -> None:
    workspace = _workspace(topk=6, max_page_table_width=5)
    q = torch.empty((4, 2, 512), dtype=torch.bfloat16)
    swa_indices = torch.empty((4, 2), dtype=torch.int32)
    swa_lengths = torch.empty((4,), dtype=torch.int32)
    indexed_indices = torch.empty((4, 4), dtype=torch.int32)
    indexed_lengths = torch.empty((4,), dtype=torch.int32)
    indexed_page_table = torch.empty((4, 5), dtype=torch.int32)

    binding = workspace.bind_compressed_mla(
        q=q,
        swa_indices=swa_indices,
        swa_lengths=swa_lengths,
        indexed_indices=indexed_indices,
        indexed_lengths=indexed_lengths,
        indexed_page_table=indexed_page_table,
    )

    assert isinstance(binding, B12XCompressedMLABinding)
    assert binding.scratch is workspace
    assert binding.q.data_ptr() == q.data_ptr()
    assert binding.indexed_page_table is indexed_page_table


def test_workspace_bind_sparse_mla_returns_common_binding_type() -> None:
    workspace = _workspace(topk=6, max_page_table_width=5)
    q = torch.empty((4, 2, 512), dtype=torch.bfloat16)
    selected_indices = torch.empty((4, 6), dtype=torch.int32)
    cache_seqlens = torch.empty((3,), dtype=torch.int32)
    active_counts = torch.empty((4,), dtype=torch.int32)

    binding = workspace.bind_sparse_mla(
        q=q,
        selected_indices=selected_indices,
        cache_seqlens_int32=cache_seqlens,
        nsa_cache_seqlens_int32=active_counts,
    )

    assert isinstance(binding, B12XSparseMLABinding)
    assert binding.workspace is workspace
    assert binding.q.data_ptr() == q.data_ptr()
    assert binding.selected_indices is selected_indices
    assert binding.cache_seqlens_int32 is cache_seqlens


def test_workspace_bind_compressed_indexer_returns_common_binding_type() -> None:
    workspace = _workspace(indexer_num_q_heads=3, max_paged_q_rows=4, max_page_table_width=7)
    real_page_table = torch.empty((4, 7), dtype=torch.int32)
    cache_seqlens = torch.empty((4,), dtype=torch.int32)
    active_width = torch.empty((1,), dtype=torch.int32)
    schedule = torch.empty((2, 2), dtype=torch.int32)

    binding = workspace.bind_compressed_indexer(
        real_page_table=real_page_table,
        cache_seqlens_int32=cache_seqlens,
        active_width=active_width,
        schedule_metadata=schedule,
        expected_num_q_heads=3,
        shared_page_table=True,
    )

    assert isinstance(binding, B12XCompressedIndexerBinding)
    assert binding.workspace is workspace
    assert binding.real_page_table is real_page_table
    assert binding.active_width is active_width
    assert binding.expected_num_q_heads == 3
    assert binding.shared_page_table is True


def test_workspace_bind_indexer_paged_decode_returns_common_binding_type() -> None:
    workspace = _workspace(indexer_num_q_heads=3, max_paged_q_rows=4, max_page_table_width=7)
    real_page_table = torch.empty((4, 7), dtype=torch.int32)
    cache_seqlens = torch.empty((4,), dtype=torch.int32)
    active_width = torch.empty((1,), dtype=torch.int32)
    schedule = torch.empty((2, 2), dtype=torch.int32)

    binding = workspace.bind_indexer_paged_decode(
        real_page_table=real_page_table,
        cache_seqlens_int32=cache_seqlens,
        active_width=active_width,
        paged_mqa_schedule_metadata=schedule,
    )

    assert isinstance(binding, B12XIndexerPagedBinding)
    assert binding.workspace is workspace
    assert binding.metadata.real_page_table is real_page_table
    assert binding.metadata.paged_mqa_schedule_metadata is schedule
    assert binding.active_width is active_width


def test_workspace_bind_indexer_extend_returns_common_binding_type() -> None:
    workspace = _workspace(indexer_num_q_heads=3, max_total_q=4, topk=8)
    workspace.indexer_extend_tile_logits = torch.empty((2048,), dtype=torch.float32)
    workspace.indexer_extend_topk_values = torch.empty((4, 8), dtype=torch.float32)
    workspace.indexer_extend_topk_indices = torch.empty((4, 8), dtype=torch.int32)
    workspace.indexer_extend_candidate_values = torch.empty((2, 4, 8), dtype=torch.float32)
    workspace.indexer_extend_candidate_indices = torch.empty((2, 4, 8), dtype=torch.int32)
    workspace.indexer_extend_topk_positions = torch.empty((4, 8), dtype=torch.int64)
    workspace.indexer_extend_lengths = torch.empty((4,), dtype=torch.int32)
    k_start = torch.zeros((3,), dtype=torch.int32)
    k_end = torch.full((3,), 64, dtype=torch.int32)

    binding = workspace.bind_indexer_extend(k_start=k_start, k_end=k_end, topk=3)

    assert isinstance(binding, B12XIndexerExtendBinding)
    assert binding.workspace is workspace
    assert binding.metadata.k_start is k_start
    assert binding.metadata.k_end is k_end
    assert binding.topk == 3
    assert binding.tile_logits is workspace.indexer_extend_tile_logits
    assert binding.output_values.shape == (3, 3)
    assert binding.output_indices.shape == (3, 3)
    assert binding.candidate_values.shape == (2, 3, 3)
    assert binding.candidate_indices.shape == (2, 3, 3)
    assert binding.merge_positions.shape == (3, 3)
    assert binding.lengths.shape == (3,)


def test_compressed_mla_decode_binding_supplies_runtime_tensors(monkeypatch) -> None:
    workspace = _workspace(max_total_q=1, topk=2, max_page_table_width=2)
    workspace.fixed_capacity = False
    workspace.use_cuda_graph = True
    workspace.tmp_output = torch.empty((1, 2, 4, 512), dtype=torch.bfloat16)
    workspace.tmp_lse = torch.empty((1, 2, 4), dtype=torch.float32)
    workspace.output_buffer = workspace.tmp_output[:, :, 0, :]
    workspace.final_lse = torch.empty((1, 2), dtype=torch.float32)
    workspace.kv_chunk_size_ptr = torch.empty((1,), dtype=torch.int32)
    workspace.num_chunks_ptr = torch.empty((1,), dtype=torch.int32)

    q = torch.zeros((1, 2, 512), dtype=torch.bfloat16)
    swa_indices = torch.zeros((1, 2), dtype=torch.int32)
    swa_lengths = torch.zeros((1,), dtype=torch.int32)
    binding = workspace.bind_compressed_mla(
        q=q,
        swa_indices=swa_indices,
        swa_lengths=swa_lengths,
    )
    swa_cache = torch.empty(
        (1, compressed_mla_page_nbytes(COMPRESSED_MLA_DSV4_PAGE_SIZE)),
        dtype=torch.uint8,
    )
    calls = {}

    def fail_stage(**kwargs):
        raise AssertionError("binding path should not stage compressed MLA inputs")

    def fake_forward(**kwargs):
        forward_binding = kwargs["binding"]
        calls["q_all"] = forward_binding.q_all
        calls["swa_indices"] = forward_binding.swa_indices
        calls["swa_lengths"] = forward_binding.swa_lengths
        forward_binding.tmp_output.zero_()

    monkeypatch.setattr(compressed_mla_impl, "_stage_fixed_compressed_mla_inputs", fail_stage)
    monkeypatch.setattr(compressed_mla_impl, "run_compressed_mla_split_decode_forward", fake_forward)

    out = compressed_mla_impl.compressed_mla_decode_forward(
        binding=binding,
        swa_k_cache=swa_cache,
        sm_scale=1.0,
    )

    assert calls["q_all"].data_ptr() == q.data_ptr()
    assert calls["swa_indices"].data_ptr() == swa_indices.data_ptr()
    assert calls["swa_lengths"].data_ptr() == swa_lengths.data_ptr()
    assert out.shape == (1, 2, 512)


def test_sparse_mla_decode_binding_supplies_runtime_tensors(monkeypatch) -> None:
    workspace = _workspace(max_total_q=1, topk=2, max_page_table_width=2)
    q = torch.zeros((1, 2, 512), dtype=torch.bfloat16)
    selected_indices = torch.zeros((1, 2), dtype=torch.int32)
    cache_seqlens = torch.zeros((1,), dtype=torch.int32)
    active_counts = torch.zeros((1,), dtype=torch.int32)
    binding = workspace.bind_sparse_mla(
        q=q,
        selected_indices=selected_indices,
        cache_seqlens_int32=cache_seqlens,
        nsa_cache_seqlens_int32=active_counts,
    )
    kv_cache = torch.empty((1, 576), dtype=torch.bfloat16)
    calls = {}

    def fake_run_sparse_mla(**kwargs):
        calls.update(kwargs)
        return torch.empty((1, 2, 512), dtype=torch.bfloat16)

    monkeypatch.setattr(sparse_mla_impl, "_run_sparse_mla", fake_run_sparse_mla)

    out = sparse_mla_impl.sparse_mla_decode_forward(
        binding=binding,
        kv_cache=kv_cache,
        sm_scale=1.0,
    )

    assert calls["q_all"].data_ptr() == q.data_ptr()
    assert calls["selected_indices"].data_ptr() == selected_indices.data_ptr()
    assert calls["cache_seqlens_int32"].data_ptr() == cache_seqlens.data_ptr()
    assert calls["active_token_counts"].data_ptr() == active_counts.data_ptr()
    assert calls["workspace"] is workspace
    assert calls["v_head_dim"] == workspace.v_head_dim
    assert out.shape == (1, 2, 512)


def test_compressed_indexer_logits_binding_supplies_metadata(monkeypatch) -> None:
    workspace = _workspace(indexer_num_q_heads=2, max_paged_q_rows=3, max_page_table_width=5)
    real_page_table = torch.zeros((3, 5), dtype=torch.int32)
    cache_seqlens = torch.zeros((3,), dtype=torch.int32)
    active_width = torch.tensor([320], dtype=torch.int32)
    binding = workspace.bind_compressed_indexer(
        real_page_table=real_page_table,
        cache_seqlens_int32=cache_seqlens,
        active_width=active_width,
        expected_num_q_heads=2,
    )
    q_fp8 = torch.empty((3, 2, 128), dtype=torch.uint8)
    weights = torch.empty((3, 2), dtype=torch.float32)
    index_k_cache = torch.empty((8, 64 * (128 + 4)), dtype=torch.uint8)
    calls = {}

    def fake_paged_decode_logits(**kwargs):
        calls.update(kwargs)
        return torch.empty((3, 320), dtype=torch.float32)

    monkeypatch.setattr(compressed_indexer_impl, "paged_decode_logits", fake_paged_decode_logits)

    logits = compressed_indexer_impl.compressed_index_decode_logits_fp8(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        binding=binding,
    )

    assert calls["metadata"].real_page_table is real_page_table
    assert calls["metadata"].cache_seqlens_int32 is cache_seqlens
    assert calls["workspace"] is workspace
    assert calls["active_width_override"] is active_width
    assert logits.shape == (3, 320)


def test_indexer_paged_decode_binding_supplies_metadata(monkeypatch) -> None:
    workspace = _workspace(indexer_num_q_heads=2, max_paged_q_rows=3, max_page_table_width=5)
    real_page_table = torch.zeros((3, 5), dtype=torch.int32)
    cache_seqlens = torch.zeros((3,), dtype=torch.int32)
    active_width = torch.tensor([0], dtype=torch.int32)
    binding = workspace.bind_indexer_paged_decode(
        real_page_table=real_page_table,
        cache_seqlens_int32=cache_seqlens,
        active_width=active_width,
    )
    q_fp8 = torch.empty((3, 2, 128), dtype=torch.uint8)
    weights = torch.empty((3, 2), dtype=torch.float32)
    index_k_cache = torch.empty((8, 64 * 128), dtype=torch.uint8)
    calls = {}

    def fake_supports(**kwargs):
        return True

    def fake_uses_schedule(**kwargs):
        return False

    def fake_run_kernel(**kwargs):
        calls.update(kwargs)
        return torch.empty((3, 320), dtype=torch.float32)

    monkeypatch.setattr(indexer_impl, "supports_paged_logits_kernel", fake_supports)
    monkeypatch.setattr(indexer_impl, "uses_paged_mqa_schedule", fake_uses_schedule)
    monkeypatch.setattr(indexer_impl, "run_paged_logits_kernel", fake_run_kernel)

    logits = indexer_impl.paged_decode_logits(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        binding=binding,
    )

    assert calls["real_page_table"] is real_page_table
    assert calls["seqlens_per_query"] is cache_seqlens
    assert calls["active_width"] is active_width
    assert calls["workspace"] is workspace
    assert logits.shape == (3, 320)


def test_indexer_extend_logits_binding_supplies_metadata(monkeypatch) -> None:
    workspace = _workspace(indexer_num_q_heads=2, max_total_q=3, topk=4)
    k_start = torch.zeros((3,), dtype=torch.int32)
    k_end = torch.full((3,), 64, dtype=torch.int32)
    binding = workspace.bind_indexer_extend(k_start=k_start, k_end=k_end, topk=2)
    q_fp8 = torch.empty((3, 2, 128), dtype=torch.uint8)
    weights = torch.empty((3, 2), dtype=torch.float32)
    k_quant = torch.empty((64, 128), dtype=torch.uint8)
    k_scale = torch.empty((64,), dtype=torch.float32)
    calls = {}

    def fake_supports(**kwargs):
        return True

    def fake_run_kernel(**kwargs):
        calls.update(kwargs)
        return torch.empty((3, 64), dtype=torch.float32)

    monkeypatch.setattr(indexer_impl, "supports_extend_logits_kernel", fake_supports)
    monkeypatch.setattr(indexer_impl, "run_extend_logits_kernel", fake_run_kernel)

    logits = indexer_impl.extend_logits(
        q_fp8=q_fp8,
        weights=weights,
        kv_fp8=(k_quant, k_scale),
        binding=binding,
    )

    assert calls["k_start"] is k_start
    assert calls["k_end"] is k_end
    assert calls["workspace"] is workspace
    assert calls["contract_phantoms"] is None
    assert logits.shape == (3, 64)


def test_indexer_extend_tiled_topk_binding_supplies_topk_and_metadata(monkeypatch) -> None:
    workspace = _workspace(indexer_num_q_heads=2, max_total_q=3, topk=4)
    k_start = torch.zeros((3,), dtype=torch.int32)
    k_end = torch.full((3,), 4, dtype=torch.int32)
    binding = workspace.bind_indexer_extend(k_start=k_start, k_end=k_end, topk=2)
    q_fp8 = torch.empty((3, 2, 128), dtype=torch.uint8)
    weights = torch.empty((3, 2), dtype=torch.float32)
    k_quant = torch.empty((4, 128), dtype=torch.uint8)
    k_scale = torch.empty((4,), dtype=torch.float32)
    logits = torch.tensor(
        [
            [0.0, 3.0, 1.0, 2.0],
            [4.0, 1.0, 2.0, 3.0],
            [1.0, 0.0, 5.0, 4.0],
        ],
        dtype=torch.float32,
    )
    calls = {}

    def fake_supports(**kwargs):
        return False

    def fake_reference(**kwargs):
        calls.update(kwargs)
        return logits

    monkeypatch.setattr(indexer_impl, "supports_extend_logits_kernel", fake_supports)
    monkeypatch.setattr(indexer_impl, "extend_logits_reference", fake_reference)

    indices = indexer_impl.extend_tiled_topk(
        q_fp8=q_fp8,
        weights=weights,
        kv_fp8=(k_quant, k_scale),
        binding=binding,
    )

    assert calls["k_start"] is k_start
    assert calls["k_end"] is k_end
    assert indices.tolist() == [[1, 3], [0, 3], [2, 3]]
