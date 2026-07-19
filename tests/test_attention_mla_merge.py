from __future__ import annotations

import math
from dataclasses import dataclass

import torch

import b12x.attention.mla.merge as mla_merge

from .helpers import require_sm120


_HEAD_DIM = 512
_LOG2_E = 1.0 / math.log(2.0)


@dataclass(frozen=True)
class _FixedMergeProblem:
    tmp_output: torch.Tensor
    tmp_lse: torch.Tensor
    num_chunks_ptr: torch.Tensor
    output: torch.Tensor


def _allocate_chunk_major_partials(
    *,
    rows: int,
    heads: int,
    chunks: int,
    device: torch.device,
) -> torch.Tensor:
    """Match the fixed workspace's [chunk][row][head][dim] physical order."""
    storage = torch.empty(
        rows * heads * chunks * _HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    return storage.as_strided(
        (rows, heads, chunks, _HEAD_DIM),
        (
            heads * _HEAD_DIM,
            _HEAD_DIM,
            rows * heads * _HEAD_DIM,
            1,
        ),
    )


def _make_fixed_merge_problem(
    *,
    rows: int,
    heads: int,
    chunks: int,
    device: torch.device,
) -> _FixedMergeProblem:
    tmp_output = _allocate_chunk_major_partials(
        rows=rows,
        heads=heads,
        chunks=chunks,
        device=device,
    )
    tmp_lse = torch.empty(
        (rows, heads, chunks),
        dtype=torch.float32,
        device=device,
    )
    num_chunks_ptr = torch.full(
        (1,),
        chunks,
        dtype=torch.int32,
        device=device,
    )

    # This is the production workspace alias: chunk zero is contiguous and is
    # reused as the final output only after the partials have been produced.
    output = tmp_output[:, :, 0, :]
    assert output.is_contiguous()
    assert output.data_ptr() == tmp_output.data_ptr()
    return _FixedMergeProblem(
        tmp_output=tmp_output,
        tmp_lse=tmp_lse,
        num_chunks_ptr=num_chunks_ptr,
        output=output,
    )


def _make_merge_scenarios(
    *,
    rows: int,
    heads: int,
    chunks: int,
    device: torch.device,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor],
]:
    generator = torch.Generator(device=device)
    generator.manual_seed(9460)

    partials_a = _allocate_chunk_major_partials(
        rows=rows,
        heads=heads,
        chunks=chunks,
        device=device,
    )
    partials_b = _allocate_chunk_major_partials(
        rows=rows,
        heads=heads,
        chunks=chunks,
        device=device,
    )
    partials_a.normal_(mean=0.0, std=0.25, generator=generator)
    partials_b.normal_(mean=0.0, std=0.25, generator=generator)

    lse_a = torch.empty(
        (rows, heads, chunks), dtype=torch.float32, device=device
    ).normal_(mean=-0.25, std=1.1, generator=generator)
    lse_b = torch.empty(
        (rows, heads, chunks), dtype=torch.float32, device=device
    ).normal_(mean=0.15, std=1.1, generator=generator)

    # Force the first-valid search, neutral chunks in the middle, and the
    # all-empty boundary. The second replay moves each boundary without
    # changing any pointer, shape, launch grid, or workspace capacity.
    lse_a[:, :, 0] = -torch.inf
    lse_a[1::2, ::4, 2] = -torch.inf
    lse_a[0, 0, :] = -torch.inf
    lse_b[:, :, 1] = -torch.inf
    lse_b[::2, 1::5, chunks - 1] = -torch.inf
    lse_b[1, 7, :] = -torch.inf
    return (partials_a, lse_a), (partials_b, lse_b)


def _split_merge_fp32_oracle(
    partials: torch.Tensor,
    lse_base2: torch.Tensor,
    *,
    chunks: int,
    attn_sink: torch.Tensor | None,
) -> torch.Tensor:
    """Independent FP32 definition of normalized-partial base-2 merging."""
    partials_fp32 = partials[:, :, :chunks, :].float()
    lse_fp32 = lse_base2[:, :, :chunks].float()
    valid = lse_fp32 != -torch.inf
    masked_lse = torch.where(valid, lse_fp32, -torch.inf)

    if attn_sink is None:
        has_partial = valid.any(dim=-1)
        merged_max = masked_lse.max(dim=-1).values
        safe_max = torch.where(has_partial, merged_max, torch.zeros_like(merged_max))
        weights = torch.where(
            valid,
            torch.exp2(lse_fp32 - safe_max.unsqueeze(-1)),
            torch.zeros_like(lse_fp32),
        )
        denominator = weights.sum(dim=-1)
        numerator = (partials_fp32 * weights.unsqueeze(-1)).sum(dim=-2)
        return torch.where(
            has_partial.unsqueeze(-1),
            numerator / denominator.clamp_min(torch.finfo(torch.float32).tiny).unsqueeze(-1),
            torch.zeros_like(numerator),
        )

    sink_lse_base2 = attn_sink.float().unsqueeze(0) * _LOG2_E
    merged_max = torch.maximum(masked_lse.max(dim=-1).values, sink_lse_base2)
    weights = torch.where(
        valid,
        torch.exp2(lse_fp32 - merged_max.unsqueeze(-1)),
        torch.zeros_like(lse_fp32),
    )
    sink_weight = torch.exp2(sink_lse_base2 - merged_max)
    denominator = weights.sum(dim=-1) + sink_weight
    numerator = (partials_fp32 * weights.unsqueeze(-1)).sum(dim=-2)
    return numerator / denominator.unsqueeze(-1)


def _install_scenario(
    problem: _FixedMergeProblem,
    *,
    partials: torch.Tensor,
    lse: torch.Tensor,
    live_sink: torch.Tensor | None,
    source_sink: torch.Tensor | None,
) -> None:
    problem.tmp_output.copy_(partials)
    problem.tmp_lse.copy_(lse)
    if live_sink is not None:
        assert source_sink is not None
        live_sink.copy_(source_sink)


def _assert_replay_matches_oracle(
    problem: _FixedMergeProblem,
    *,
    graph: torch.cuda.CUDAGraph,
    partials: torch.Tensor,
    lse: torch.Tensor,
    expected: torch.Tensor,
    live_sink: torch.Tensor | None,
    source_sink: torch.Tensor | None,
) -> None:
    _install_scenario(
        problem,
        partials=partials,
        lse=lse,
        live_sink=live_sink,
        source_sink=source_sink,
    )
    graph.replay()
    torch.cuda.synchronize(problem.output.device)

    torch.testing.assert_close(
        problem.output.float(),
        expected,
        atol=1.5e-2,
        rtol=1.5e-2,
    )
    all_empty = (lse == -torch.inf).all(dim=-1)
    assert torch.count_nonzero(problem.output[all_empty]).item() == 0
    assert torch.count_nonzero(problem.output[~all_empty]).item() > 0


def _run_fixed_merge_graph_case(*, with_sink: bool) -> None:
    device = require_sm120()
    rows = 4
    heads = 32
    chunks = 5
    problem = _make_fixed_merge_problem(
        rows=rows,
        heads=heads,
        chunks=chunks,
        device=device,
    )
    scenario_a, scenario_b = _make_merge_scenarios(
        rows=rows,
        heads=heads,
        chunks=chunks,
        device=device,
    )

    sink_a: torch.Tensor | None = None
    sink_b: torch.Tensor | None = None
    live_sink: torch.Tensor | None = None
    if with_sink:
        sink_a = torch.linspace(-1.25, 0.75, heads, dtype=torch.float32, device=device)
        sink_b = torch.linspace(0.9, -0.6, heads, dtype=torch.float32, device=device)
        live_sink = torch.empty_like(sink_a)

    expected_a = _split_merge_fp32_oracle(
        scenario_a[0],
        scenario_a[1],
        chunks=chunks,
        attn_sink=sink_a,
    )
    expected_b = _split_merge_fp32_oracle(
        scenario_b[0],
        scenario_b[1],
        chunks=chunks,
        attn_sink=sink_b,
    )
    binding = mla_merge.build_sparse_mla_split_decode_merge_binding(
        tmp_output=problem.tmp_output,
        tmp_lse=problem.tmp_lse,
        num_chunks_ptr=problem.num_chunks_ptr,
        output=problem.output,
        num_chunks=chunks,
        attn_sink=live_sink,
    )

    _install_scenario(
        problem,
        partials=scenario_a[0],
        lse=scenario_a[1],
        live_sink=live_sink,
        source_sink=sink_a,
    )
    binding.run()
    torch.cuda.synchronize(device)

    # Compilation and all allocations happen before capture. Only the fixed
    # binding launch is captured; replay-time inputs are copied into the same
    # preplanned storage from tensors allocated above.
    _install_scenario(
        problem,
        partials=scenario_a[0],
        lse=scenario_a[1],
        live_sink=live_sink,
        source_sink=sink_a,
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        binding.run()

    _assert_replay_matches_oracle(
        problem,
        graph=graph,
        partials=scenario_a[0],
        lse=scenario_a[1],
        expected=expected_a,
        live_sink=live_sink,
        source_sink=sink_a,
    )
    _assert_replay_matches_oracle(
        problem,
        graph=graph,
        partials=scenario_b[0],
        lse=scenario_b[1],
        expected=expected_b,
        live_sink=live_sink,
        source_sink=sink_b,
    )


@torch.inference_mode()
def test_sparse_mla_split_decode_merge_graph_replay_matches_fp32_oracle() -> None:
    _run_fixed_merge_graph_case(with_sink=False)


@torch.inference_mode()
def test_sparse_mla_split_decode_sink_merge_graph_replay_matches_fp32_oracle() -> None:
    _run_fixed_merge_graph_case(with_sink=True)
