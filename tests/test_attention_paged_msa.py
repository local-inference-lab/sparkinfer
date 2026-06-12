from __future__ import annotations

import math

import torch

from b12x.attention.paged.reference import (
    materialize_paged_kv_cache,
    msa_attention_reference,
)
from b12x.integration.attention import (
    B12XPagedAttentionScratchCaps,
    clear_attention_caches,
    create_paged_plan,
    paged_attention_forward,
    plan_paged_attention_scratch,
)

from .helpers import require_sm120
from .paged_attention_helpers import (
    make_msa_q2k_indices,
    make_paged_inputs,
    quantize_paged_kv_cache_e4m3,
)


def _msa_dense_mask_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    q2k_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    total_q, q_heads, head_dim = q.shape
    kv_heads = k_cache.shape[2]
    q_per_kv = q_heads // kv_heads
    out = torch.empty(
        (total_q, q_heads, v_cache.shape[-1]), dtype=q.dtype, device=q.device
    )
    lse = torch.empty((total_q, q_heads), dtype=torch.float32, device=q.device)
    q_offsets = [int(v) for v in cu_seqlens_q.detach().cpu().tolist()]
    scale = head_dim ** -0.5

    for request_idx, (q_start, q_end) in enumerate(zip(q_offsets[:-1], q_offsets[1:])):
        cache_len = int(cache_seqlens[request_idx].item())
        qo_len = q_end - q_start
        k_full, v_full = materialize_paged_kv_cache(
            k_cache,
            v_cache,
            page_table,
            cache_seqlens,
            request_idx=request_idx,
        )
        for q_row in range(q_start, q_end):
            token_local = q_row - q_start
            causal_limit = token_local + cache_len - qo_len
            for q_head in range(q_heads):
                kv_head = q_head // q_per_kv
                scores = (
                    torch.matmul(
                        k_full[:, kv_head].to(torch.float32),
                        q[q_row, q_head].to(torch.float32),
                    )
                    * scale
                )
                mask = torch.ones((cache_len,), dtype=torch.bool, device=q.device)
                for block_id_raw in q2k_indices[kv_head, q_row].detach().cpu().tolist():
                    block_id = int(block_id_raw)
                    if block_id < 0:
                        continue
                    start = block_id * 128
                    end = min(start + 128, causal_limit + 1, cache_len)
                    if end > start:
                        mask[start:end] = False
                scores = scores.masked_fill(mask, float("-inf"))
                probs = torch.softmax(scores, dim=0)
                out[q_row, q_head].copy_(
                    torch.matmul(probs, v_full[:, kv_head].to(torch.float32)).to(q.dtype)
                )
                lse[q_row, q_head] = torch.logsumexp(scores, dim=0)
    return out, lse


def _assert_close(out: torch.Tensor, ref: torch.Tensor, lse: torch.Tensor, ref_lse: torch.Tensor) -> None:
    torch.testing.assert_close(lse, ref_lse, rtol=0, atol=1e-5)
    cosine = torch.nn.functional.cosine_similarity(
        out.to(torch.float32).reshape(-1),
        ref.to(torch.float32).reshape(-1),
        dim=0,
    ).item()
    assert cosine >= 0.99999


def _run_msa_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    q2k_indices: torch.Tensor,
    *,
    fixed_split_size: int | None = None,
    k_descale: torch.Tensor | None = None,
    v_descale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    plan = create_paged_plan(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        mode="decode",
        msa_block_sparse=True,
        fixed_split_size=-1 if fixed_split_size is None else fixed_split_size,
    )
    assert plan.split_kv is True
    assert plan.kv_chunk_size % 64 == 0
    assert 64 <= plan.kv_chunk_size <= 2048
    assert plan.total_num_partial_rows >= plan.new_batch_size
    scratch_plan = plan_paged_attention_scratch(
        B12XPagedAttentionScratchCaps(
            device=q.device,
            mode="decode",
            dtype=q.dtype,
            kv_dtype=k_cache.dtype,
            num_q_heads=q.shape[1],
            num_kv_heads=k_cache.shape[2],
            head_dim_qk=q.shape[2],
            head_dim_vo=v_cache.shape[3],
            page_size=k_cache.shape[1],
            max_total_q=plan.total_q,
            max_batch=page_table.shape[0],
            max_page_table_width=page_table.shape[1],
            max_work_items=max(plan.padded_batch_size, 1),
            max_partial_rows=plan.total_num_partial_rows,
            num_cache_pages=k_cache.shape[0],
            msa_block_sparse=True,
        )
    )
    scratch = tuple(
        torch.empty(shape, dtype=dtype, device=q.device)
        for shape, dtype in scratch_plan.shapes_and_dtypes()
    )
    output = torch.empty(
        (q.shape[0], q.shape[1], v_cache.shape[3]), dtype=q.dtype, device=q.device
    )
    binding = scratch_plan.bind(
        scratch=scratch,
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        output=output,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        fixed_split_size=fixed_split_size,
        q2k_indices=q2k_indices,
        k_descale=k_descale,
        v_descale=v_descale,
    )
    out, lse_base2 = paged_attention_forward(binding=binding)
    return out, _lse_base2_to_natural(lse_base2)


def _run_msa_extend(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    q2k_indices: torch.Tensor,
    *,
    k_descale: torch.Tensor | None = None,
    v_descale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    plan = create_paged_plan(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        mode="extend",
        msa_block_sparse=True,
    )
    assert plan.split_kv is False
    assert plan.cta_tile_q in (16, 128)
    assert plan.new_batch_size > 0
    scratch_plan = plan_paged_attention_scratch(
        B12XPagedAttentionScratchCaps(
            device=q.device,
            mode="extend",
            dtype=q.dtype,
            kv_dtype=k_cache.dtype,
            num_q_heads=q.shape[1],
            num_kv_heads=k_cache.shape[2],
            head_dim_qk=q.shape[2],
            head_dim_vo=v_cache.shape[3],
            page_size=k_cache.shape[1],
            max_total_q=plan.total_q,
            max_batch=page_table.shape[0],
            max_page_table_width=page_table.shape[1],
            max_work_items=max(plan.new_batch_size, 1),
            max_partial_rows=0,
            num_cache_pages=k_cache.shape[0],
            msa_block_sparse=True,
        )
    )
    scratch = tuple(
        torch.empty(shape, dtype=dtype, device=q.device)
        for shape, dtype in scratch_plan.shapes_and_dtypes()
    )
    output = torch.empty(
        (q.shape[0], q.shape[1], v_cache.shape[3]), dtype=q.dtype, device=q.device
    )
    binding = scratch_plan.bind(
        scratch=scratch,
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        output=output,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        q2k_indices=q2k_indices,
        k_descale=k_descale,
        v_descale=v_descale,
    )
    out, lse_base2 = paged_attention_forward(binding=binding)
    return out, _lse_base2_to_natural(lse_base2)


def test_msa_attention_reference_matches_dense_mask_small_decode() -> None:
    require_sm120()
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = make_paged_inputs(
        q_seqlens=[1, 1],
        cache_seqlens=[129, 513],
        page_size=64,
        q_heads=64,
        kv_heads=4,
        head_dim=128,
        dtype=torch.bfloat16,
        seed=1201,
    )
    q2k_indices = make_msa_q2k_indices(
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        num_kv_heads=4,
        seed=7,
        force_block0=True,
    )

    out, lse = msa_attention_reference(
        q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, q2k_indices
    )
    ref, ref_lse = _msa_dense_mask_reference(
        q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, q2k_indices
    )
    _assert_close(out, ref, lse, ref_lse)


def test_msa_attention_reference_ignores_poisoned_padding() -> None:
    require_sm120()
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = make_paged_inputs(
        q_seqlens=[1],
        cache_seqlens=[127],
        page_size=64,
        q_heads=64,
        kv_heads=4,
        head_dim=128,
        dtype=torch.bfloat16,
        seed=1202,
    )
    clean = make_msa_q2k_indices(
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        num_kv_heads=4,
        seed=11,
    )
    poisoned = make_msa_q2k_indices(
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        num_kv_heads=4,
        seed=11,
        poison_padding=True,
    )

    out, lse = msa_attention_reference(
        q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, clean
    )
    poison_out, poison_lse = msa_attention_reference(
        q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, poisoned
    )
    _assert_close(poison_out, out, poison_lse, lse)


def test_msa_attention_reference_handles_varlen_extend_causality() -> None:
    require_sm120()
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = make_paged_inputs(
        q_seqlens=[5, 3],
        cache_seqlens=[200, 384],
        page_size=64,
        q_heads=64,
        kv_heads=4,
        head_dim=128,
        dtype=torch.bfloat16,
        seed=1203,
    )
    q2k_indices = make_msa_q2k_indices(
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        num_kv_heads=4,
        total_q_capacity=16,
        seed=19,
        force_block0=True,
    )

    out, lse = msa_attention_reference(
        q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, q2k_indices
    )
    ref, ref_lse = _msa_dense_mask_reference(
        q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, q2k_indices
    )
    _assert_close(out, ref, lse, ref_lse)


def _lse_base2_to_natural(lse: torch.Tensor) -> torch.Tensor:
    return lse * math.log(2.0)


def test_msa_decode_eager_bf16_matches_reference_tail_cases() -> None:
    require_sm120()
    cache_lens = [1, 64, 127, 128, 129, 200, 2047, 2048, 5000]
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = make_paged_inputs(
        q_seqlens=[1] * len(cache_lens),
        cache_seqlens=cache_lens,
        page_size=64,
        q_heads=64,
        kv_heads=4,
        head_dim=128,
        dtype=torch.bfloat16,
        seed=1301,
    )
    q2k_indices = make_msa_q2k_indices(
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        num_kv_heads=4,
        seed=31,
        force_block0=True,
        poison_padding=True,
    )

    out, lse = _run_msa_decode(
        q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, q2k_indices
    )
    ref, ref_lse = msa_attention_reference(
        q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, q2k_indices
    )
    torch.testing.assert_close(lse, ref_lse, rtol=2e-3, atol=2e-3)
    cosine = torch.nn.functional.cosine_similarity(
        out.to(torch.float32).reshape(-1),
        ref.to(torch.float32).reshape(-1),
        dim=0,
    ).item()
    assert cosine >= 0.999


def test_msa_decode_eager_bf16_split_chunk_invariance() -> None:
    require_sm120()
    cache_lens = [1, 64, 127, 128, 129, 200, 2047, 2048, 5000]
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = make_paged_inputs(
        q_seqlens=[1] * len(cache_lens),
        cache_seqlens=cache_lens,
        page_size=64,
        q_heads=64,
        kv_heads=4,
        head_dim=128,
        dtype=torch.bfloat16,
        seed=1302,
    )
    q2k_indices = make_msa_q2k_indices(
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        num_kv_heads=4,
        seed=37,
        force_block0=True,
        poison_padding=True,
    )
    ref, ref_lse = msa_attention_reference(
        q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, q2k_indices
    )

    outputs: list[torch.Tensor] = []
    lses: list[torch.Tensor] = []
    for fixed_split_size in (1, 2, 4, 32):
        out, lse = _run_msa_decode(
            q,
            k_cache,
            v_cache,
            page_table,
            cache_seqlens,
            cu_seqlens_q,
            q2k_indices,
            fixed_split_size=fixed_split_size,
        )
        torch.testing.assert_close(lse, ref_lse, rtol=2e-3, atol=2e-3)
        cosine = torch.nn.functional.cosine_similarity(
            out.to(torch.float32).reshape(-1),
            ref.to(torch.float32).reshape(-1),
            dim=0,
        ).item()
        assert cosine >= 0.999
        outputs.append(out.detach().clone())
        lses.append(lse.detach().clone())

    for out in outputs[1:]:
        torch.testing.assert_close(out, outputs[0], rtol=3e-3, atol=3e-3)
    for lse in lses[1:]:
        torch.testing.assert_close(lse, lses[0], rtol=2e-3, atol=2e-3)


def test_msa_decode_eager_fp8_kv_matches_reference() -> None:
    require_sm120()
    cache_lens = [129, 2048, 5000]
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = make_paged_inputs(
        q_seqlens=[1] * len(cache_lens),
        cache_seqlens=cache_lens,
        page_size=64,
        q_heads=64,
        kv_heads=4,
        head_dim=128,
        dtype=torch.bfloat16,
        seed=1303,
    )
    q2k_indices = make_msa_q2k_indices(
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        num_kv_heads=4,
        seed=41,
        force_block0=True,
        poison_padding=True,
    )
    k_fp8, v_fp8, k_descale, v_descale = quantize_paged_kv_cache_e4m3(
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
    )

    out, lse = _run_msa_decode(
        q,
        k_fp8,
        v_fp8,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        q2k_indices,
        k_descale=k_descale,
        v_descale=v_descale,
    )
    ref, ref_lse = msa_attention_reference(
        q,
        k_fp8,
        v_fp8,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        q2k_indices,
        k_descale=k_descale,
        v_descale=v_descale,
    )
    torch.testing.assert_close(lse, ref_lse, rtol=5e-2, atol=5e-2)
    cosine = torch.nn.functional.cosine_similarity(
        out.to(torch.float32).reshape(-1),
        ref.to(torch.float32).reshape(-1),
        dim=0,
    ).item()
    assert cosine >= 0.995


def test_msa_extend_eager_bf16_matches_reference_varlen() -> None:
    require_sm120()
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = make_paged_inputs(
        q_seqlens=[1, 5, 300],
        cache_seqlens=[129, 384, 640],
        page_size=64,
        q_heads=64,
        kv_heads=4,
        head_dim=128,
        dtype=torch.bfloat16,
        seed=1304,
    )
    q2k_indices = make_msa_q2k_indices(
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        num_kv_heads=4,
        seed=43,
        force_block0=True,
        poison_padding=True,
    )

    out, lse = _run_msa_extend(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        q2k_indices,
    )
    ref, ref_lse = msa_attention_reference(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        q2k_indices,
    )
    torch.testing.assert_close(lse, ref_lse, rtol=2e-3, atol=2e-3)
    cosine = torch.nn.functional.cosine_similarity(
        out.to(torch.float32).reshape(-1),
        ref.to(torch.float32).reshape(-1),
        dim=0,
    ).item()
    assert cosine >= 0.999


def test_msa_extend_qo_len_one_matches_decode() -> None:
    require_sm120()
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = make_paged_inputs(
        q_seqlens=[1, 1],
        cache_seqlens=[129, 2048],
        page_size=64,
        q_heads=64,
        kv_heads=4,
        head_dim=128,
        dtype=torch.bfloat16,
        seed=1305,
    )
    q2k_indices = make_msa_q2k_indices(
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        num_kv_heads=4,
        seed=47,
        force_block0=True,
    )

    extend_out, extend_lse = _run_msa_extend(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        q2k_indices,
    )
    decode_out, decode_lse = _run_msa_decode(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        q2k_indices,
    )
    torch.testing.assert_close(extend_lse, decode_lse, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(extend_out, decode_out, rtol=3e-3, atol=3e-3)


@torch.inference_mode()
def test_msa_decode_cuda_graph_replays_with_mutating_metadata_and_q2k() -> None:
    require_sm120()
    clear_attention_caches()

    batch = 2
    page_table_width = 80
    num_pages = 512

    def make_case(cache_lens: list[int], *, seed: int):
        return make_paged_inputs(
            q_seqlens=[1] * batch,
            cache_seqlens=cache_lens,
            page_size=64,
            q_heads=64,
            kv_heads=4,
            head_dim=128,
            dtype=torch.bfloat16,
            seed=seed,
            page_table_width=page_table_width,
            num_pages=num_pages,
        )

    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = make_case(
        [2048, 5000],
        seed=1401,
    )
    q2k_indices = make_msa_q2k_indices(
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        num_kv_heads=4,
        total_q_capacity=batch,
        seed=1402,
        force_block0=True,
    )
    q2k_data_ptr = int(q2k_indices.data_ptr())

    scratch_plan = plan_paged_attention_scratch(
        B12XPagedAttentionScratchCaps(
            device=q.device,
            mode="decode",
            dtype=q.dtype,
            kv_dtype=k_cache.dtype,
            num_q_heads=q.shape[1],
            num_kv_heads=k_cache.shape[2],
            head_dim_qk=q.shape[2],
            head_dim_vo=v_cache.shape[3],
            page_size=k_cache.shape[1],
            max_total_q=batch,
            max_batch=batch,
            max_page_table_width=page_table_width,
            max_work_items=batch * 32,
            max_partial_rows=batch * 32,
            num_cache_pages=num_pages,
            use_cuda_graph=True,
            msa_block_sparse=True,
        )
    )
    scratch_plan.prepare_decode_graph_replay_state(
        batch=batch,
        max_page_table_width=page_table_width,
        max_cache_page_count=page_table_width,
    )
    scratch = tuple(
        torch.empty(shape, dtype=dtype, device=q.device)
        for shape, dtype in scratch_plan.shapes_and_dtypes()
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
        q2k_indices=q2k_indices,
    )

    out, lse = paged_attention_forward(binding=binding)
    torch.cuda.synchronize()
    ref_out, ref_lse = msa_attention_reference(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        q2k_indices,
    )
    torch.testing.assert_close(lse * math.log(2.0), ref_lse, rtol=2e-3, atol=2e-3)
    assert (
        torch.nn.functional.cosine_similarity(
            out.to(torch.float32).reshape(-1),
            ref_out.to(torch.float32).reshape(-1),
            dim=0,
        ).item()
        >= 0.999
    )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        paged_attention_forward(binding=binding)

    graph.replay()
    torch.cuda.synchronize()
    lse_view = binding.scratch.current_lse_view() * math.log(2.0)
    torch.testing.assert_close(lse_view, ref_lse, rtol=2e-3, atol=2e-3)
    assert (
        torch.nn.functional.cosine_similarity(
            output.to(torch.float32).reshape(-1),
            ref_out.to(torch.float32).reshape(-1),
            dim=0,
        ).item()
        >= 0.999
    )

    q_next, k_next, v_next, page_table_next, cache_seqlens_next, cu_seqlens_q_next = (
        make_case([1, 129], seed=1403)
    )
    q2k_next = make_msa_q2k_indices(
        cache_seqlens=cache_seqlens_next,
        cu_seqlens_q=cu_seqlens_q_next,
        num_kv_heads=4,
        total_q_capacity=batch,
        seed=1404,
        force_block0=True,
    )
    q.copy_(q_next)
    k_cache.copy_(k_next)
    v_cache.copy_(v_next)
    q2k_indices.copy_(q2k_next)
    assert int(q2k_indices.data_ptr()) == q2k_data_ptr
    binding = scratch_plan.bind(
        scratch=scratch,
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        output=output,
        page_table=page_table_next,
        cache_seqlens=cache_seqlens_next,
        cu_seqlens_q=cu_seqlens_q_next,
        q2k_indices=q2k_indices,
    )

    graph.replay()
    torch.cuda.synchronize()
    ref_out_next, ref_lse_next = msa_attention_reference(
        q,
        k_cache,
        v_cache,
        page_table_next,
        cache_seqlens_next,
        cu_seqlens_q_next,
        q2k_indices,
    )
    lse_next = binding.scratch.current_lse_view() * math.log(2.0)
    torch.testing.assert_close(lse_next, ref_lse_next, rtol=2e-3, atol=2e-3)
    assert (
        torch.nn.functional.cosine_similarity(
            output.to(torch.float32).reshape(-1),
            ref_out_next.to(torch.float32).reshape(-1),
            dim=0,
        ).item()
        >= 0.999
    )
