#!/usr/bin/env python3
"""Exact-object same-address CUDA-graph ABBA for NSA/indexer SM120 kernels.

The thirteen cases are the uncovered migration exception rows: MSA contiguous
block scores; NSA contiguous decode, prefill, prefill-512, and tiled prefill;
the four persistent-top-k row specializations; and paged tiled,
scheduled-single, scheduled-multi, and streamed scoring.
Every launch uses the production binding/API and is pinned to exact cache
objects with a Torch oracle. Captured serving inputs are mutated in place at
stable addresses after timing to prove that replay consumes live values.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType
from typing import Any

import torch

from benchmarks.common import make_l2_flush_fn
from validation.cutlass_migration.core.evidence_status import (
    add_evidence_status_argument,
)
from validation.cutlass_migration.core.exact_cache_abba import (
    allocator_counters,
    gpu_mode_snapshot,
    graph_topology,
    load_exact,
    pin_module_launches,
    require_target_gpu,
    sha256_file,
    tensor_sha256,
    time_conditions,
    topology_signature,
    verify_artifact,
)
from validation.cutlass_migration.paths import CORE_ROOT, REPO_ROOT
from sparkinfer.attention.indexer import (
    MSA_SM_SCALE,
    build_paged_mqa_schedule_metadata,
    clear_indexer_caches,
    contiguous_tiled_topk,
)
import sparkinfer.attention.indexer.contiguous_kernel as contiguous_kernel
from sparkinfer.attention.indexer.contiguous_kernel import (
    run_contiguous_block_scores_kernel,
)
import sparkinfer.attention.indexer.kernel as paged_kernel
from sparkinfer.attention.indexer.kernel import (
    build_indexer_paged_logits_kernel_binding,
    build_indexer_paged_supertile_logits_kernel_binding,
    build_indexer_paged_tiled_logits_kernel_binding,
)
import sparkinfer.attention.indexer.persistent_topk as persistent_topk_kernel
from sparkinfer.attention.indexer.persistent_topk import (
    persistent_topk2048_scratch_nbytes,
    run_persistent_topk2048,
)
from sparkinfer.attention.indexer.msa_reference import (
    msa_contiguous_block_scores_reference,
)
from sparkinfer.attention.indexer.reference import (
    contiguous_logits_reference,
)
from sparkinfer.attention.indexer.scratch import (
    SPARKINFERIndexerContiguousScratchCaps,
    plan_indexer_contiguous_scratch,
)
import sparkinfer.attention.indexer.tiled_topk as tiled_topk_kernel
from tests.test_attention_msa_indexer_api import (
    _make_msa_contiguous_case,
    _one_scratch,
)
from tests.test_attention_nsa_indexer_api import (
    _bind_contiguous_topk,
    _bind_staged_contiguous_logits,
    _quantize_rows_to_kv_fp8,
)
from tests.test_cute_migration_nsa_paged_corpus import (
    _capture_prototype,
    _copy_paged_scenario,
    _make_paged_scenarios,
    _packed_k_cache,
    _paged_reference,
)


_TILE_Q = 32
_TILE_K = 512

_SPEC_MSA_BLOCK = "192540422d6b74e00db7c15f7fac1a5c9672153d8d75c4d7f24edcfbdd27f692"
_SPEC_CONTIG_DECODE = "517cd51e328d79745afc5d5aafbcc74bcabfd215c1b4014d6521f2f726720d54"
_SPEC_CONTIG_PREFILL = (
    "236964e34344b2d0c350d25c55cbc8410df803dac8c5b9d039f7c887bc648395"
)
_SPEC_CONTIG_PREFILL512 = (
    "829e1e1a13cd41d31274a3cfdff01d52d92ba5187989b495d26947ecb68fec02"
)
_SPEC_CONTIG_TILED = "9333322e702d20e062008e7eece2ed2031ecbc4fa9043b9625d28c738a20b0d6"
_SPEC_TILED_TOPK = "5d7340996bb1b2d036544fe7c8af9f6168c18eef48550a542a2961f3bda91479"
_SPEC_PAGED_BASE = "6cf90208cbb6373504c59da1bfe2d9bd49e31c3cbd2c46b9ad4bfa54707816a7"
_SPEC_PAGED_SINGLE = "9b974da7d3d2089edb2a56acf1ac902e3dde68bbd97a7a6722ae2b8e5eb75aee"
_SPEC_PAGED_MULTI = "01fdad91dca0e1c16cac35242077aa74e983e16b8414ec17d3e2a38c2a5409e8"
_SPEC_PAGED_STREAM = "2d50ff677a56cde4b45e81a1ae68e4ddb18ef6951d5736d4098d32aecd9a94e5"
_SPEC_PERSISTENT_TOPK_ROWS = {
    1: "4a9976060484a3f7e53212152205039fdb1ef25622ff721a2f0513c7423c80fd",
    2: "4c130e75383f6867d2a43a6c852989f751aeb72898dbf5690039e725adc32529",
    3: "b7cc1e9a311dba01484625236d3ff055c80a9dd5bb7775efd25f34b2a7a5f846",
    4: "0da4241709d1d1f59a59b6d7fa5532c03008e4fdd1979736443f78425718d6ab",
}


@dataclass(frozen=True)
class CaseDefinition:
    name: str
    target_specs: tuple[str, ...]
    all_specs: tuple[str, ...]


_DEFINITIONS = {
    case.name: case
    for case in (
        CaseDefinition(
            "msa-contiguous-block-scores", (_SPEC_MSA_BLOCK,), (_SPEC_MSA_BLOCK,)
        ),
        CaseDefinition(
            "contiguous-decode", (_SPEC_CONTIG_DECODE,), (_SPEC_CONTIG_DECODE,)
        ),
        CaseDefinition(
            "contiguous-prefill", (_SPEC_CONTIG_PREFILL,), (_SPEC_CONTIG_PREFILL,)
        ),
        CaseDefinition(
            "contiguous-prefill512-h32",
            (_SPEC_CONTIG_PREFILL512,),
            (_SPEC_CONTIG_PREFILL512,),
        ),
        CaseDefinition(
            "contiguous-tiled-prefill",
            (_SPEC_CONTIG_TILED,),
            (_SPEC_CONTIG_TILED, _SPEC_TILED_TOPK),
        ),
        *(
            CaseDefinition(
                f"persistent-topk-rows{rows}",
                (spec_hash,),
                (spec_hash,),
            )
            for rows, spec_hash in _SPEC_PERSISTENT_TOPK_ROWS.items()
        ),
        CaseDefinition("paged-base-tiled", (_SPEC_PAGED_BASE,), (_SPEC_PAGED_BASE,)),
        CaseDefinition(
            "paged-scheduled-single",
            (_SPEC_PAGED_SINGLE,),
            (_SPEC_PAGED_SINGLE,),
        ),
        CaseDefinition(
            "paged-scheduled-multi",
            (_SPEC_PAGED_MULTI,),
            (_SPEC_PAGED_MULTI,),
        ),
        CaseDefinition("paged-stream", (_SPEC_PAGED_STREAM,), (_SPEC_PAGED_STREAM,)),
    )
}


@dataclass
class RuntimeCase:
    definition: CaseDefinition
    modules: Mapping[ModuleType, tuple[str, ...]]
    launch: Callable[[], object]
    poison: Callable[[], None]
    validate: Callable[[], dict[str, object]]
    snapshot: Callable[[], Mapping[str, torch.Tensor]]
    install_scenario: Callable[[int], None]
    live_inputs: Mapping[str, torch.Tensor]
    read_only: Mapping[str, torch.Tensor]
    stable_tensors: Mapping[str, torch.Tensor]
    launch_context: Callable[[], Any] = nullcontext
    contract: Mapping[str, object] | None = None


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_evidence_status_argument(parser)
    parser.add_argument("--a-cache", type=Path, required=True)
    parser.add_argument("--a-label", default="cutlass-4.5.2")
    parser.add_argument("--b-cache", type=Path, required=True)
    parser.add_argument("--b-label", default="cutlass-4.6.0")
    parser.add_argument("--case", choices=tuple(_DEFINITIONS), required=True)
    parser.add_argument("--precondition", type=int, default=400)
    parser.add_argument("--precondition-seconds", type=float, default=5.0)
    parser.add_argument("--maximum-precondition-seconds", type=float, default=60.0)
    parser.add_argument("--max-sm-clock-delta-mhz", type=float, default=60.0)
    parser.add_argument("--warmup", type=int, default=80)
    parser.add_argument("--cycles", type=int, default=500)
    parser.add_argument("--event-batch-cycles", type=int, default=50)
    parser.add_argument(
        "--cold-l2", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--l2-flush-bytes", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _package_versions() -> dict[str, str]:
    versions = {}
    for package in (
        "nvidia-cutlass-dsl",
        "nvidia-cutlass-dsl-libs-base",
        "nvidia-cutlass-dsl-libs-core",
        "nvidia-cutlass-dsl-libs-cu12",
        "nvidia-cutlass-dsl-libs-cu13",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "missing"
    return versions


def _tensor_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, object]:
    actual_f32 = actual.float()
    expected_f32 = expected.float()
    torch.testing.assert_close(actual_f32, expected_f32, atol=1e-4, rtol=1e-4)
    expected_finite = torch.isfinite(expected_f32)
    actual_finite = torch.isfinite(actual_f32)
    if not torch.equal(actual_finite, expected_finite):
        raise AssertionError("actual and oracle finite-value masks differ")
    difference = torch.zeros_like(actual_f32)
    difference[expected_finite] = (
        actual_f32[expected_finite] - expected_f32[expected_finite]
    )
    return {
        "shape": list(actual.shape),
        "finite": True,
        "expected_nonfinite": int(torch.count_nonzero(~expected_finite).item()),
        "nonzero": int(torch.count_nonzero(actual_f32[actual_finite]).item()),
        "max_abs": float(difference.abs().max().item()),
        "sha256": tensor_sha256(actual),
    }


def _build_contiguous_logits(
    definition: CaseDefinition,
    *,
    q_rows: int,
    heads: int,
    k_rows: int,
    prefill_block_k: int | None,
) -> RuntimeCase:
    device = torch.device("cuda")
    generator = torch.Generator(device="cpu").manual_seed(
        20260719 + q_rows * 31 + heads * 17 + k_rows
    )
    q_fp8 = (
        torch.randn((q_rows, heads, 128), generator=generator, dtype=torch.float32)
        .div_(2)
        .to(device=device, dtype=torch.float8_e4m3fn)
    )
    weights = torch.randn((q_rows, heads), generator=generator, dtype=torch.float32).to(
        device=device
    )
    k = (
        torch.randn((k_rows, 128), generator=generator, dtype=torch.float32)
        .div_(3)
        .to(device=device)
    )
    kv_fp8 = _quantize_rows_to_kv_fp8(k)
    k_start = torch.zeros(q_rows, dtype=torch.int32, device=device)
    k_end = torch.full((q_rows,), k_rows, dtype=torch.int32, device=device)
    binding, output, bound_kv = _bind_staged_contiguous_logits(
        q_fp8=q_fp8,
        weights=weights,
        kv_fp8=kv_fp8,
        k_start=k_start,
        k_end=k_end,
        prefill_block_k=prefill_block_k,
    )
    k_quant, k_scale = bound_kv
    sampled_q = torch.tensor(
        sorted({0, q_rows // 2, q_rows - 1}), dtype=torch.long, device=device
    )
    sampled_k = torch.tensor(
        sorted({1, 2, k_rows // 2, k_rows - 2}), dtype=torch.long, device=device
    )
    live_inputs = {
        "q_fp8": q_fp8,
        "weights": weights,
        "k_start": k_start,
        "k_end": k_end,
    }
    scenarios = (
        {name: tensor.clone() for name, tensor in live_inputs.items()},
        {
            "q_fp8": (-q_fp8.float()).to(torch.float8_e4m3fn),
            "weights": weights.float().mul(-0.75).add(0.03125),
            "k_start": torch.ones_like(k_start),
            "k_end": torch.full_like(k_end, k_rows - 1),
        },
    )

    def install_scenario(index: int) -> None:
        for name, tensor in live_inputs.items():
            tensor.copy_(scenarios[index][name])

    def poison() -> None:
        output.fill_(float("nan"))

    def checked() -> torch.Tensor:
        return output[sampled_q[:, None], sampled_k[None, :]]

    def validate() -> dict[str, object]:
        actual = checked()
        reference = contiguous_logits_reference(
            q_fp8=q_fp8,
            weights=weights,
            kv_fp8=bound_kv,
            k_start=k_start,
            k_end=k_end,
        )
        expected = reference[sampled_q[:, None], sampled_k[None, :]]
        metrics = _tensor_metrics(actual, expected)
        if int(metrics["nonzero"]) == 0:
            raise AssertionError("contiguous scorer produced only zeros")
        return {
            "sampled_logits": metrics,
            "checked_region_finite": bool(torch.isfinite(actual).all()),
        }

    read_only = {
        "k_quant": k_quant,
        "k_scale": k_scale,
    }
    return RuntimeCase(
        definition=definition,
        modules={contiguous_kernel: definition.all_specs},
        launch=binding.run,
        poison=poison,
        validate=validate,
        snapshot=lambda: {"sampled_logits": checked()},
        install_scenario=install_scenario,
        live_inputs=live_inputs,
        read_only=read_only,
        stable_tensors={**live_inputs, **read_only, "output": output},
        contract={
            "q_rows": q_rows,
            "heads": heads,
            "k_rows": k_rows,
            "prefill_block_k": prefill_block_k,
            "oracle": "sampled independent FP32 scorer",
        },
    )


def _build_msa_block_scores(definition: CaseDefinition) -> RuntimeCase:
    device = torch.device("cuda")
    rows, heads, k_rows = 257, 4, 384
    q_fp8, q_scale, kv_fp8, metadata = _make_msa_contiguous_case(
        rows=rows,
        heads=heads,
        k_rows=k_rows,
        device=device,
        seed=20260719,
    )
    plan = plan_indexer_contiguous_scratch(
        SPARKINFERIndexerContiguousScratchCaps(
            device=device,
            num_q_heads=1,
            num_idx_heads=heads,
            max_q_rows=rows,
            max_k_rows=k_rows,
            topk=16,
            score_mode="msa",
        )
    )
    plan_binding = plan.bind_msa(
        scratch=_one_scratch(plan),
        k_start=metadata.k_start,
        k_end=metadata.k_end,
    )
    scratch = plan_binding.scratch
    k_quant, k_scale = kv_fp8
    scratch.k_quant[:k_rows].copy_(k_quant)
    scratch.k_scale[:k_rows].copy_(k_scale)
    scratch.prepare_k_padding(k_rows=k_rows)
    bound_k_quant = scratch.k_quant[:k_rows]
    bound_k_scale = scratch.k_scale[:k_rows]
    weights = (q_scale * MSA_SM_SCALE).contiguous()
    q_bytes = q_fp8.view(torch.uint8)
    q_u32 = q_bytes.view(torch.uint32).view(rows, heads, 32)
    if plan_binding.block_scores is None:
        raise RuntimeError("MSA scratch plan did not allocate block scores")
    block_scores = plan_binding.block_scores
    live_inputs = {
        "q_fp8": q_fp8,
        "k_start": metadata.k_start,
        "k_end": metadata.k_end,
    }
    shift_forward = metadata.k_end < k_rows
    scenario_1_start = torch.where(
        shift_forward,
        metadata.k_start + 1,
        (metadata.k_start - 1).clamp_min(0),
    )
    scenario_1_end = torch.where(
        shift_forward,
        metadata.k_end + 1,
        metadata.k_end - 1,
    )
    scenarios = (
        {name: tensor.clone() for name, tensor in live_inputs.items()},
        {
            "q_fp8": (-q_fp8.float()).to(torch.float8_e4m3fn),
            "k_start": scenario_1_start,
            "k_end": scenario_1_end,
        },
    )

    def install_scenario(index: int) -> None:
        for name, tensor in live_inputs.items():
            tensor.copy_(scenarios[index][name])

    def launch() -> torch.Tensor:
        return run_contiguous_block_scores_kernel(
            q_fp8=q_fp8,
            weights=weights,
            k_quant=bound_k_quant,
            k_scale=bound_k_scale,
            k_start=metadata.k_start,
            k_end=metadata.k_end,
            block_scores=block_scores,
            num_blocks_out=int(block_scores.shape[2]),
            q_u32=q_u32,
            q_bytes=q_bytes,
            weights_kernel=weights,
            k_quant_bytes=scratch.k_quant.view(torch.uint8),
            k_scale_kernel=scratch.k_scale,
            k_start_kernel=metadata.k_start,
            k_end_kernel=metadata.k_end,
            out_kernel=scratch.dummy_logits,
            tile_logits_kernel=scratch.tile_logits,
            k_tma_prefill_desc_ptrs=scratch.k_tma_prefill_desc_ptrs,
        )

    def validate() -> dict[str, object]:
        expected = msa_contiguous_block_scores_reference(
            q_fp8=q_fp8,
            q_scale=q_scale,
            kv_fp8=(bound_k_quant, bound_k_scale),
            k_start=metadata.k_start,
            k_end=metadata.k_end,
        )
        metrics = _tensor_metrics(block_scores, expected)
        if int(metrics["nonzero"]) == 0:
            raise AssertionError("MSA block scorer produced only zeros")
        return {"block_scores": metrics}

    read_only = {
        "q_scale": q_scale,
        "weights": weights,
        "k_quant": bound_k_quant,
        "k_scale": bound_k_scale,
    }
    return RuntimeCase(
        definition=definition,
        modules={contiguous_kernel: definition.all_specs},
        launch=launch,
        poison=lambda: block_scores.fill_(float("nan")),
        validate=validate,
        snapshot=lambda: {"block_scores": block_scores},
        install_scenario=install_scenario,
        live_inputs=live_inputs,
        read_only=read_only,
        stable_tensors={
            **live_inputs,
            **read_only,
            "block_scores": block_scores,
            "scratch": plan_binding.scratch.shared_scratch,
        },
        contract={
            "rows": rows,
            "heads": heads,
            "k_rows": k_rows,
            "score_mode": "msa",
            "oracle": "msa_contiguous_block_scores_reference",
        },
    )


def _build_contiguous_tiled(definition: CaseDefinition) -> RuntimeCase:
    device = torch.device("cuda")
    q_rows, heads, k_rows, topk = 32, 8, 1024, 512
    generator = torch.Generator(device="cpu").manual_seed(20260720)
    q_fp8 = (
        torch.randn((q_rows, heads, 128), generator=generator, dtype=torch.float32)
        .div_(2)
        .to(device=device, dtype=torch.float8_e4m3fn)
    )
    weights = torch.randn((q_rows, heads), generator=generator, dtype=torch.float32).to(
        device=device
    )
    k = (
        torch.randn((k_rows, 128), generator=generator, dtype=torch.float32)
        .div_(3)
        .to(device=device)
    )
    kv_fp8 = _quantize_rows_to_kv_fp8(k)
    k_start = torch.zeros(q_rows, dtype=torch.int32, device=device)
    k_end = torch.full((q_rows,), k_rows, dtype=torch.int32, device=device)
    binding, bound_kv = _bind_contiguous_topk(
        kv_fp8=kv_fp8,
        k_start=k_start,
        k_end=k_end,
        num_q_heads=heads,
        topk=topk,
        q_rows=q_rows,
        supertile_k=k_rows,
    )
    if binding.output_indices is None or binding.output_values is None:
        raise RuntimeError("contiguous top-k binding did not allocate outputs")
    live_inputs = {
        "q_fp8": q_fp8,
        "weights": weights,
        "k_start": k_start,
        "k_end": k_end,
    }
    scenarios = (
        {name: tensor.clone() for name, tensor in live_inputs.items()},
        {
            "q_fp8": (-q_fp8.float()).to(torch.float8_e4m3fn),
            "weights": weights.float().mul(-0.75).add(0.03125),
            "k_start": torch.ones_like(k_start),
            "k_end": torch.full_like(k_end, k_rows - 1),
        },
    )

    def install_scenario(index: int) -> None:
        for name, tensor in live_inputs.items():
            tensor.copy_(scenarios[index][name])

    def launch() -> torch.Tensor:
        return contiguous_tiled_topk(
            q_fp8=q_fp8,
            weights=weights,
            kv_fp8=bound_kv,
            binding=binding,
        )

    def poison() -> None:
        binding.output_indices.fill_(-1)
        binding.output_values.fill_(float("nan"))

    def validate() -> dict[str, object]:
        logits = contiguous_logits_reference(
            q_fp8=q_fp8,
            weights=weights,
            kv_fp8=bound_kv,
            k_start=k_start,
            k_end=k_end,
        )
        expected_values = torch.topk(
            logits,
            k=topk,
            dim=1,
            largest=True,
            sorted=False,
        ).values
        if bool((binding.output_indices < 0).any()) or bool(
            (binding.output_indices >= k_rows).any()
        ):
            raise AssertionError("contiguous tiled top-k left invalid indices")
        gathered = torch.gather(logits, 1, binding.output_indices.long())
        torch.testing.assert_close(
            binding.output_values.float(), gathered.float(), atol=1e-4, rtol=1e-4
        )
        actual = torch.sort(binding.output_values, dim=1).values
        expected = torch.sort(expected_values, dim=1).values
        metrics = _tensor_metrics(actual, expected)
        if int(metrics["nonzero"]) == 0:
            raise AssertionError("contiguous tiled top-k left poisoned outputs")
        return {
            "topk_values": metrics,
            "topk_equal": True,
            "indices_sha256": tensor_sha256(binding.output_indices),
        }

    k_quant, k_scale = bound_kv
    read_only = {
        "k_quant": k_quant,
        "k_scale": k_scale,
    }
    return RuntimeCase(
        definition=definition,
        modules={
            contiguous_kernel: (_SPEC_CONTIG_TILED,),
            tiled_topk_kernel: (_SPEC_TILED_TOPK,),
        },
        launch=launch,
        poison=poison,
        validate=validate,
        snapshot=lambda: {
            "indices": binding.output_indices,
            "values": binding.output_values,
        },
        install_scenario=install_scenario,
        live_inputs=live_inputs,
        read_only=read_only,
        stable_tensors={
            **live_inputs,
            **read_only,
            "output_indices": binding.output_indices,
            "output_values": binding.output_values,
            "scratch": binding.scratch.shared_scratch,
        },
        contract={
            "q_rows": q_rows,
            "heads": heads,
            "k_rows": k_rows,
            "topk": topk,
            "production_tiled_topk_route": True,
            "support_spec": _SPEC_TILED_TOPK,
            "oracle": "contiguous_logits_reference + torch.topk",
        },
    )


def _valid_paged_snapshot(
    actual: torch.Tensor,
    seqlens: torch.Tensor,
) -> torch.Tensor:
    return torch.cat(
        [actual[row, : int(seqlens[row].item())] for row in range(seqlens.numel())]
    )


def _build_paged_tiled(
    definition: CaseDefinition,
    *,
    stream: bool,
) -> RuntimeCase:
    device = torch.device("cuda")
    generator = torch.Generator(device="cpu").manual_seed(20260721 + int(stream))
    if stream:
        rows, heads, max_pages, cache_pages = 2, 64, 8, 96
        lengths_a, lengths_b = (509, 443), (251, 187)
    else:
        rows, heads, max_pages, cache_pages = 4, 8, 8, 96
        lengths_a, lengths_b = (509, 447, 383, 319), (257, 191, 127, 63)
    index_k_cache = _packed_k_cache(
        cache_pages,
        generator=generator,
        device=device,
    )
    scenarios = _make_paged_scenarios(
        rows=rows,
        heads=heads,
        max_pages=max_pages,
        cache_pages=cache_pages,
        lengths_a=lengths_a,
        lengths_b=lengths_b,
        generator=generator,
        device=device,
    )
    q_fp8 = torch.empty_like(scenarios[0][0])
    weights = torch.empty_like(scenarios[0][1])
    page_table = torch.empty_like(scenarios[0][2])
    seqlens = torch.empty_like(scenarios[0][3])
    active_width = torch.empty((1,), dtype=torch.int32, device=device)
    tile_logits = torch.empty((_TILE_Q * _TILE_K,), dtype=torch.float32, device=device)
    _copy_paged_scenario(
        q_fp8=q_fp8,
        weights=weights,
        page_table=page_table,
        seqlens=seqlens,
        active_width=active_width,
        scenario=_capture_prototype(scenarios[0]),
    )
    if stream:
        binding = build_indexer_paged_supertile_logits_kernel_binding(
            q_fp8=q_fp8,
            weights=weights,
            index_k_cache=index_k_cache,
            real_page_table=page_table,
            seqlens_per_query=seqlens,
            active_width=active_width,
            tile_logits=tile_logits,
            source_page_offset=0,
            output_width_tokens=_TILE_K,
            tile_block_q=_TILE_Q,
            tile_block_k=_TILE_K,
            preinitialize_tile_logits=False,
        )
    else:
        binding = build_indexer_paged_tiled_logits_kernel_binding(
            q_fp8=q_fp8,
            weights=weights,
            index_k_cache=index_k_cache,
            real_page_table=page_table,
            seqlens_per_query=seqlens,
            active_width=active_width,
            tile_logits=tile_logits,
            tile_block_q=_TILE_Q,
            tile_block_k=_TILE_K,
            preinitialize_tile_logits=False,
        )
    actual = tile_logits.view(_TILE_Q, _TILE_K)[:rows]
    live_inputs = {
        "q_fp8": q_fp8,
        "weights": weights,
        "page_table": page_table,
        "seqlens": seqlens,
        "active_width": active_width,
    }

    def install_scenario(index: int) -> None:
        _copy_paged_scenario(
            q_fp8=q_fp8,
            weights=weights,
            page_table=page_table,
            seqlens=seqlens,
            active_width=active_width,
            scenario=scenarios[index],
        )

    def validate() -> dict[str, object]:
        expected = _paged_reference(
            q_fp8=q_fp8,
            weights=weights,
            index_k_cache=index_k_cache,
            page_table=page_table,
            seqlens=seqlens,
        )
        actual_valid = _valid_paged_snapshot(actual, seqlens)
        expected_valid = _valid_paged_snapshot(expected, seqlens)
        metrics = _tensor_metrics(actual_valid, expected_valid)
        if int(metrics["nonzero"]) == 0:
            raise AssertionError("paged tiled scorer produced only zeros")
        return {"valid_logits": metrics, "poison_contract": "live seqlen regions"}

    read_only = {"index_k_cache": index_k_cache}
    return RuntimeCase(
        definition=definition,
        modules={paged_kernel: definition.all_specs},
        launch=binding.run,
        poison=lambda: tile_logits.fill_(float("nan")),
        validate=validate,
        snapshot=lambda: {"valid_logits": _valid_paged_snapshot(actual, seqlens)},
        install_scenario=install_scenario,
        live_inputs=live_inputs,
        read_only=read_only,
        stable_tensors={**live_inputs, **read_only, "tile_logits": tile_logits},
        contract={
            "rows": rows,
            "heads": heads,
            "max_pages": max_pages,
            "cache_pages": cache_pages,
            "stream_supertile": stream,
            "oracle": "paged_decode_logits_reference",
        },
    )


@contextmanager
def _stable_empty_output(output: torch.Tensor) -> Iterator[None]:
    original = torch.empty
    expected_shape = tuple(output.shape)

    def stable_empty(*args, **kwargs):
        shape = tuple(args[0]) if args and isinstance(args[0], (tuple, list)) else None
        dtype = kwargs.get("dtype")
        device = torch.device(kwargs.get("device", "cpu"))
        if shape == expected_shape and dtype == output.dtype and device.type == "cuda":
            return output
        return original(*args, **kwargs)

    torch.empty = stable_empty
    try:
        yield
    finally:
        torch.empty = original


def _build_paged_scheduled(
    definition: CaseDefinition,
    *,
    rows: int,
) -> RuntimeCase:
    device = torch.device("cuda")
    generator = torch.Generator(device="cpu").manual_seed(20260722 + rows)
    heads, max_pages, cache_pages = 8, 1024, 160
    lengths_a = (4093,) if rows == 1 else (4093, 3581)
    lengths_b = (3071,) if rows == 1 else (2815, 3327)
    index_k_cache = _packed_k_cache(
        cache_pages,
        generator=generator,
        device=device,
    )
    scenarios = _make_paged_scenarios(
        rows=rows,
        heads=heads,
        max_pages=max_pages,
        cache_pages=cache_pages,
        lengths_a=lengths_a,
        lengths_b=lengths_b,
        generator=generator,
        device=device,
    )
    q_fp8 = torch.empty_like(scenarios[0][0])
    weights = torch.empty_like(scenarios[0][1])
    page_table = torch.empty_like(scenarios[0][2])
    seqlens = torch.empty_like(scenarios[0][3])
    active_width = torch.empty((1,), dtype=torch.int32, device=device)
    schedule = torch.empty((9, 2), dtype=torch.int32, device=device)
    _copy_paged_scenario(
        q_fp8=q_fp8,
        weights=weights,
        page_table=page_table,
        seqlens=seqlens,
        active_width=active_width,
        scenario=_capture_prototype(scenarios[0]),
    )
    build_paged_mqa_schedule_metadata(seqlens, 64, 8, out=schedule)
    output = torch.empty((rows, max_pages * 64), dtype=torch.float32, device=device)
    binding = build_indexer_paged_logits_kernel_binding(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        real_page_table=page_table,
        seqlens_per_query=seqlens,
        schedule_metadata=schedule,
        active_width=active_width,
        preinitialize_invalid_logits=False,
    )
    live_inputs = {
        "q_fp8": q_fp8,
        "weights": weights,
        "page_table": page_table,
        "seqlens": seqlens,
        "active_width": active_width,
        "schedule": schedule,
    }

    def install_scenario(index: int) -> None:
        _copy_paged_scenario(
            q_fp8=q_fp8,
            weights=weights,
            page_table=page_table,
            seqlens=seqlens,
            active_width=active_width,
            scenario=scenarios[index],
        )
        build_paged_mqa_schedule_metadata(seqlens, 64, 8, out=schedule)

    def validate() -> dict[str, object]:
        expected = _paged_reference(
            q_fp8=q_fp8,
            weights=weights,
            index_k_cache=index_k_cache,
            page_table=page_table,
            seqlens=seqlens,
        )
        actual_valid = _valid_paged_snapshot(output, seqlens)
        expected_valid = _valid_paged_snapshot(expected, seqlens)
        metrics = _tensor_metrics(actual_valid, expected_valid)
        if int(metrics["nonzero"]) == 0:
            raise AssertionError("scheduled paged scorer produced only zeros")
        return {"valid_logits": metrics, "poison_contract": "live seqlen regions"}

    read_only = {"index_k_cache": index_k_cache}
    return RuntimeCase(
        definition=definition,
        modules={paged_kernel: definition.all_specs},
        launch=binding.run,
        poison=lambda: output.fill_(float("nan")),
        validate=validate,
        snapshot=lambda: {"valid_logits": _valid_paged_snapshot(output, seqlens)},
        install_scenario=install_scenario,
        live_inputs=live_inputs,
        read_only=read_only,
        stable_tensors={**live_inputs, **read_only, "output": output},
        launch_context=lambda: _stable_empty_output(output),
        contract={
            "rows": rows,
            "heads": heads,
            "max_pages": max_pages,
            "cache_pages": cache_pages,
            "caller_stable_output": True,
            "preinitialize_invalid_logits": False,
            "oracle": "paged_decode_logits_reference",
        },
    )


def _build_persistent_topk(
    definition: CaseDefinition,
    *,
    rows: int,
) -> RuntimeCase:
    device = torch.device("cuda")
    width, topk = 33_792, 512
    generator = torch.Generator(device="cpu").manual_seed(84_000 + rows)
    scenario_logits = tuple(
        torch.randn((2, rows, width), generator=generator, dtype=torch.float32)
        .to(device=device)
        .unbind(0)
    )
    row_offsets = torch.arange(rows, dtype=torch.int32, device=device)
    scenario_lengths = (
        width - 1 - row_offsets * 53,
        width - 17 - torch.flip(row_offsets, dims=(0,)) * 67,
    )
    logical = torch.arange(width, dtype=torch.int32, device=device)
    scenario_tables = tuple(
        torch.stack(
            tuple(
                torch.roll(logical, shifts=scenario * 97 + row * 31)
                + scenario * 1_000_000
                + row * 100_000
                for row in range(rows)
            )
        )
        for scenario in range(2)
    )
    logits = torch.empty_like(scenario_logits[0])
    lengths = torch.empty_like(scenario_lengths[0])
    page_table = torch.empty_like(scenario_tables[0])
    output = torch.empty((rows, topk), dtype=torch.int32, device=device)
    scratch_nbytes = persistent_topk2048_scratch_nbytes(rows, width, device=device)
    int32_nbytes = torch.empty((), dtype=torch.int32).element_size()
    if scratch_nbytes % int32_nbytes:
        raise RuntimeError("persistent top-k scratch is not int32 aligned")
    scratch = torch.empty(
        (scratch_nbytes // int32_nbytes,),
        dtype=torch.int32,
        device=device,
    )
    live_inputs = {
        "logits": logits,
        "lengths": lengths,
        "page_table": page_table,
    }
    read_only = {
        "scenario_0_logits": scenario_logits[0],
        "scenario_1_logits": scenario_logits[1],
        "scenario_0_lengths": scenario_lengths[0],
        "scenario_1_lengths": scenario_lengths[1],
        "scenario_0_page_table": scenario_tables[0],
        "scenario_1_page_table": scenario_tables[1],
    }

    def install_scenario(index: int) -> None:
        logits.copy_(scenario_logits[index])
        lengths.copy_(scenario_lengths[index])
        page_table.copy_(scenario_tables[index])

    def launch() -> torch.Tensor:
        return run_persistent_topk2048(
            logits,
            lengths,
            page_table_1=page_table,
            output_indices=output,
            scratch=scratch,
            max_seq_len=width,
            topk=topk,
        )

    def sorted_output() -> torch.Tensor:
        return torch.sort(output, dim=1).values

    def validate() -> dict[str, object]:
        columns = torch.arange(width, dtype=torch.int64, device=device)
        masked = torch.where(
            columns.unsqueeze(0) < lengths.unsqueeze(1),
            logits,
            torch.full_like(logits, float("-inf")),
        )
        selected = torch.topk(masked, topk, dim=1, largest=True, sorted=False)
        expected_indices = torch.gather(page_table, 1, selected.indices)
        if not bool(torch.isfinite(selected.values).all().item()):
            raise AssertionError(
                "persistent top-k reference selected non-finite values"
            )
        if not bool((selected.values != 0).any().item()):
            raise AssertionError("persistent top-k reference selected only zeros")
        if bool((output < 0).any().item()):
            raise AssertionError("persistent top-k left poisoned output indices")
        actual_sorted = sorted_output()
        expected_sorted = torch.sort(expected_indices, dim=1).values
        torch.testing.assert_close(actual_sorted, expected_sorted, rtol=0.0, atol=0.0)
        return {
            "indices_sha256": tensor_sha256(actual_sorted),
            "all_rows_exact": True,
            "rows": rows,
            "width": width,
            "topk": topk,
        }

    return RuntimeCase(
        definition=definition,
        modules={persistent_topk_kernel: definition.all_specs},
        launch=launch,
        poison=lambda: output.fill_(-1),
        validate=validate,
        snapshot=lambda: {"sorted_indices": sorted_output()},
        install_scenario=install_scenario,
        live_inputs=live_inputs,
        read_only=read_only,
        stable_tensors={
            **live_inputs,
            **read_only,
            "output": output,
            "scratch": scratch,
        },
        contract={
            "rows": rows,
            "width": width,
            "topk": topk,
            "flat_1d_runtime_layout": True,
            "shared_code_resource_group": "persistent_topk_rows1_4_cutlass_460",
            "oracle": "torch.topk with exact per-row page-table remap",
        },
    )


def _build_runtime(name: str) -> RuntimeCase:
    definition = _DEFINITIONS[name]
    if name == "msa-contiguous-block-scores":
        return _build_msa_block_scores(definition)
    if name == "contiguous-decode":
        return _build_contiguous_logits(
            definition,
            q_rows=5,
            heads=3,
            k_rows=64,
            prefill_block_k=None,
        )
    if name == "contiguous-prefill":
        return _build_contiguous_logits(
            definition,
            q_rows=256,
            heads=8,
            k_rows=512,
            prefill_block_k=256,
        )
    if name == "contiguous-prefill512-h32":
        return _build_contiguous_logits(
            definition,
            q_rows=1024,
            heads=32,
            k_rows=4096,
            prefill_block_k=512,
        )
    if name == "contiguous-tiled-prefill":
        return _build_contiguous_tiled(definition)
    if name.startswith("persistent-topk-rows"):
        return _build_persistent_topk(
            definition,
            rows=int(name.removeprefix("persistent-topk-rows")),
        )
    if name == "paged-base-tiled":
        return _build_paged_tiled(definition, stream=False)
    if name == "paged-scheduled-single":
        return _build_paged_scheduled(definition, rows=1)
    if name == "paged-scheduled-multi":
        return _build_paged_scheduled(definition, rows=2)
    if name == "paged-stream":
        return _build_paged_tiled(definition, stream=True)
    raise AssertionError(name)


def _snapshot_sha256(snapshot: Mapping[str, torch.Tensor]) -> str:
    tensor_hashes = {
        name: tensor_sha256(tensor) for name, tensor in sorted(snapshot.items())
    }
    payload = json.dumps(tensor_hashes, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _capture_arm(
    *,
    label: str,
    runtime: RuntimeCase,
    compiled: Mapping[str, object],
    stream: torch.cuda.Stream,
) -> tuple[torch.cuda.CUDAGraph, list[str]]:
    observed: list[str] = []
    with ExitStack() as stack:
        for module, specs in runtime.modules.items():
            stack.enter_context(
                pin_module_launches(
                    module,
                    {spec: compiled[spec] for spec in specs},
                    observed,
                )
            )
        stack.enter_context(runtime.launch_context())
        with torch.cuda.stream(stream):
            runtime.launch()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph, stream=stream):
            runtime.launch()
        stream.synchronize()
    expected = set(runtime.definition.all_specs)
    if not observed or set(observed) != expected:
        raise RuntimeError(
            f"{label} production route observed {sorted(set(observed))}, "
            f"expected {sorted(expected)}"
        )
    return graph, observed


@torch.inference_mode()
def _run(args: argparse.Namespace) -> dict[str, object]:
    gpu = require_target_gpu()
    expected_physical_gpu = int(gpu["physical_index"])
    gpu_mode_initial = gpu_mode_snapshot(expected_physical_gpu)
    labels = (args.a_label, args.b_label)
    if labels[0] == labels[1]:
        raise ValueError("A/B labels must differ")
    if min(args.precondition, args.warmup, args.event_batch_cycles) <= 0:
        raise ValueError("precondition, warmup, and event batch must be positive")
    if args.precondition_seconds < 5.0:
        raise ValueError("release timing requires --precondition-seconds >= 5")
    if (
        args.maximum_precondition_seconds < args.precondition_seconds
        or args.maximum_precondition_seconds > 60.0
    ):
        raise ValueError(
            "release timing requires precondition-seconds <= "
            "maximum-precondition-seconds <= 60"
        )
    if args.max_sm_clock_delta_mhz != 60.0:
        raise ValueError("release timing requires --max-sm-clock-delta-mhz 60")
    if args.cycles < 500:
        raise ValueError("--cycles must be at least 500")
    if not args.cold_l2:
        raise ValueError("release ABBA evidence requires warm- and cold-L2 timing")
    definition = _DEFINITIONS[args.case]
    caches = {labels[0]: args.a_cache.resolve(), labels[1]: args.b_cache.resolve()}
    compiled: dict[str, dict[str, object]] = {label: {} for label in labels}
    artifacts: dict[str, dict[str, dict[str, Any]]] = {label: {} for label in labels}
    for label in labels:
        for spec_hash in definition.all_specs:
            exact, provenance = load_exact(caches[label], spec_hash)
            compiled[label][spec_hash] = exact
            artifacts[label][spec_hash] = provenance
    artifact_verification_before = {
        label: {
            spec_hash: verify_artifact(provenance)
            for spec_hash, provenance in artifacts[label].items()
        }
        for label in labels
    }
    for spec_hash in definition.all_specs:
        a = artifacts[labels[0]][spec_hash]
        b = artifacts[labels[1]][spec_hash]
        if a["compile_spec_json"] != b["compile_spec_json"]:
            raise RuntimeError(f"A/B compile specifications differ for {spec_hash}")
        if a["kernel_id"] != b["kernel_id"]:
            raise RuntimeError(f"A/B kernel ids differ for {spec_hash}")

    clear_indexer_caches()
    runtime = _build_runtime(args.case)
    if set(runtime.definition.all_specs) != set(definition.all_specs):
        raise AssertionError("runtime case/spec contract drifted")
    fixed_pointers_before = {
        name: tensor.data_ptr() for name, tensor in runtime.stable_tensors.items()
    }
    read_only_sha256_before = {
        name: tensor_sha256(tensor) for name, tensor in runtime.read_only.items()
    }
    stream = torch.cuda.Stream()
    graphs: dict[str, torch.cuda.CUDAGraph] = {}
    observed_specs: dict[str, list[str]] = {}
    for label in labels:
        graph, observed = _capture_arm(
            label=label,
            runtime=runtime,
            compiled=compiled[label],
            stream=stream,
        )
        graphs[label] = graph
        observed_specs[label] = observed
    topologies = {label: graph_topology(graph) for label, graph in graphs.items()}
    topology_equal = topology_signature(topologies[labels[0]]) == topology_signature(
        topologies[labels[1]]
    )
    if not topology_equal:
        raise AssertionError("A/B CUDA graph topology differs")

    allocator_checks: list[dict[str, object]] = []

    def replay_scenario(
        scenario: str,
    ) -> tuple[
        dict[str, dict[str, object]],
        dict[str, dict[str, torch.Tensor]],
    ]:
        scenario_correctness: dict[str, dict[str, object]] = {}
        snapshots: dict[str, dict[str, torch.Tensor]] = {}
        for label in labels:
            with torch.cuda.stream(stream):
                runtime.poison()
            stream.synchronize()
            counters_before = allocator_counters()
            with torch.cuda.stream(stream):
                graphs[label].replay()
            stream.synchronize()
            counters_after = allocator_counters()
            if counters_before != counters_after:
                raise AssertionError(
                    f"CUDA allocator state changed during {scenario} {label} replay"
                )
            allocator_checks.append(
                {
                    "scenario": scenario,
                    "label": label,
                    "before": counters_before,
                    "after": counters_after,
                    "unchanged": True,
                }
            )
            scenario_correctness[label] = runtime.validate()
            snapshots[label] = {
                name: tensor.clone() for name, tensor in runtime.snapshot().items()
            }
        if snapshots[labels[0]].keys() != snapshots[labels[1]].keys():
            raise AssertionError(
                f"A/B correctness snapshots differ in structure for {scenario}"
            )
        for name in snapshots[labels[0]]:
            if not torch.equal(snapshots[labels[0]][name], snapshots[labels[1]][name]):
                raise AssertionError(
                    f"A/B checked output {name} is not bit exact for {scenario}"
                )
        return scenario_correctness, snapshots

    runtime.install_scenario(0)
    torch.cuda.synchronize()
    scenario_0_sha256 = {
        name: tensor_sha256(tensor) for name, tensor in runtime.live_inputs.items()
    }
    correctness_scenario_0_pre, scenario_0_snapshots_pre = replay_scenario(
        "scenario_0_pre"
    )

    # Allocate the fixed cold-L2 sweep before the replay-allocation baseline.
    if args.cold_l2:
        make_l2_flush_fn(True, args.l2_flush_bytes)
    allocation_before = allocator_counters()
    conditions = time_conditions(
        graphs,
        labels=labels,
        precondition=args.precondition,
        warmup=args.warmup,
        cycles=args.cycles,
        event_batch_cycles=args.event_batch_cycles,
        stream=stream,
        cold_l2=args.cold_l2,
        l2_flush_bytes=args.l2_flush_bytes,
        precondition_seconds=args.precondition_seconds,
        maximum_precondition_seconds=args.maximum_precondition_seconds,
        mode_snapshot=lambda: gpu_mode_snapshot(expected_physical_gpu),
        required_pstate="P1",
        max_sm_clock_delta_mhz=args.max_sm_clock_delta_mhz,
    )
    allocation_after = allocator_counters()
    if allocation_before != allocation_after:
        raise AssertionError(
            "CUDA allocator state changed across graph replay timing: "
            f"before={allocation_before}, after={allocation_after}"
        )
    scenario_0_sha256_after_timing = {
        name: tensor_sha256(tensor) for name, tensor in runtime.live_inputs.items()
    }
    if scenario_0_sha256_after_timing != scenario_0_sha256:
        raise AssertionError("scenario-0 live inputs changed during timing")
    correctness_scenario_0_post, scenario_0_snapshots_post = replay_scenario(
        "scenario_0_post"
    )
    for label in labels:
        for name in scenario_0_snapshots_pre[label]:
            if not torch.equal(
                scenario_0_snapshots_pre[label][name],
                scenario_0_snapshots_post[label][name],
            ):
                raise AssertionError(
                    f"{label} scenario-0 {name} changed across timing interval"
                )

    runtime.install_scenario(1)
    torch.cuda.synchronize()
    scenario_1_sha256 = {
        name: tensor_sha256(tensor) for name, tensor in runtime.live_inputs.items()
    }
    changed_inputs = {
        name: scenario_0_sha256[name] != scenario_1_sha256[name]
        for name in runtime.live_inputs
    }
    if not changed_inputs or not all(changed_inputs.values()):
        raise AssertionError(
            f"scenario-1 did not mutate every captured live input: {changed_inputs}"
        )
    correctness_scenario_1, scenario_1_snapshots = replay_scenario(
        "scenario_1_live_mutation"
    )
    changed_outputs = {
        label: any(
            not torch.equal(
                scenario_0_snapshots_post[label][name],
                scenario_1_snapshots[label][name],
            )
            for name in scenario_0_snapshots_post[label]
        )
        for label in labels
    }
    if not all(changed_outputs.values()):
        raise AssertionError(
            f"live input mutation did not change every arm output: {changed_outputs}"
        )

    read_only_sha256_after = {
        name: tensor_sha256(tensor) for name, tensor in runtime.read_only.items()
    }
    if read_only_sha256_before != read_only_sha256_after:
        raise AssertionError("read-only inputs changed")
    fixed_pointers_after = {
        name: tensor.data_ptr() for name, tensor in runtime.stable_tensors.items()
    }
    if fixed_pointers_before != fixed_pointers_after:
        raise AssertionError("captured input/output/workspace addresses changed")
    artifact_verification_after = {
        label: {
            spec_hash: verify_artifact(provenance)
            for spec_hash, provenance in artifacts[label].items()
        }
        for label in labels
    }
    if artifact_verification_after != artifact_verification_before:
        raise RuntimeError("exact cache artifacts changed during benchmark")
    gpu_mode_final = gpu_mode_snapshot(expected_physical_gpu)

    return {
        "schema": "sparkinfer.attention.indexer.exact_cache_abba.v1",
        "evidence_status": args.evidence_status,
        "case": {
            "name": definition.name,
            "target_spec_hashes": definition.target_specs,
            "all_graph_spec_hashes": definition.all_specs,
            **dict(runtime.contract or {}),
        },
        "labels": {"a": labels[0], "b": labels[1]},
        "gpu": gpu,
        "gpu_mode_initial": gpu_mode_initial,
        "gpu_mode_final": gpu_mode_final,
        "provenance": {
            "command": [str(Path(sys.executable).resolve()), *sys.argv],
            "cwd": os.getcwd(),
            "git_commit": _git_output("rev-parse", "HEAD"),
            "git_worktree": _git_output("rev-parse", "--show-toplevel"),
            "git_status_short": _git_output("status", "--short").splitlines(),
            "source_sha256": {
                "benchmark": sha256_file(Path(__file__).resolve()),
                "shared_abba": sha256_file(CORE_ROOT / "exact_cache_abba.py"),
                "contiguous_kernel": sha256_file(
                    REPO_ROOT / "sparkinfer/attention/indexer/contiguous_kernel.py"
                ),
                "paged_kernel": sha256_file(
                    REPO_ROOT / "sparkinfer/attention/indexer/kernel.py"
                ),
                "persistent_topk": sha256_file(
                    REPO_ROOT / "sparkinfer/attention/indexer/persistent_topk.py"
                ),
                "tiled_topk": sha256_file(
                    REPO_ROOT / "sparkinfer/attention/indexer/tiled_topk.py"
                ),
                "compiler": sha256_file(REPO_ROOT / "sparkinfer/cute/compiler.py"),
            },
            "packages": _package_versions(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
        "artifacts": artifacts,
        "artifact_verification_before": artifact_verification_before,
        "artifact_verification_after": artifact_verification_after,
        "observed_compile_specs": observed_specs,
        "production_bindings_or_api": True,
        "exact_cache_objects": True,
        "no_recompile": True,
        "cuda_graph_replay": True,
        "cuda_graph_topology": topologies,
        "cuda_graph_topology_equal": topology_equal,
        "fixed_workspace_capacity": True,
        "fixed_allocation": True,
        "same_address_arms": True,
        "fixed_pointers": {
            "before": fixed_pointers_before,
            "after": fixed_pointers_after,
        },
        "read_only_inputs_unchanged": True,
        "read_only_inputs": {
            "unchanged": True,
            "sha256_before": read_only_sha256_before,
            "sha256_after": read_only_sha256_after,
            "timed_live_scenario_0": {
                "unchanged": True,
                "sha256_before": scenario_0_sha256,
                "sha256_after": scenario_0_sha256_after_timing,
            },
        },
        "live_input_scenarios_distinct": True,
        "live_input_mutation_changed_input": True,
        "live_input_mutation_changed_output": True,
        "live_input_mutation": {
            "captured_graph_reused": True,
            "in_place": True,
            "same_addresses": True,
            "scenarios_distinct": True,
            "changed_input": True,
            "changed_output": True,
            "scenario_0_sha256": scenario_0_sha256,
            "scenario_1_sha256": scenario_1_sha256,
            "changed_inputs": changed_inputs,
            "changed_outputs": changed_outputs,
            "scenario_0_output_sha256": {
                label: _snapshot_sha256(snapshots)
                for label, snapshots in scenario_0_snapshots_post.items()
            },
            "scenario_1_output_sha256": {
                label: _snapshot_sha256(snapshots)
                for label, snapshots in scenario_1_snapshots.items()
            },
            "scenario_0_output_tensor_sha256": {
                label: {
                    name: tensor_sha256(tensor) for name, tensor in snapshots.items()
                }
                for label, snapshots in scenario_0_snapshots_post.items()
            },
            "scenario_1_output_tensor_sha256": {
                label: {
                    name: tensor_sha256(tensor) for name, tensor in snapshots.items()
                }
                for label, snapshots in scenario_1_snapshots.items()
            },
            "allocation_addresses_stable": True,
        },
        "poisoned_checked_regions_overwritten": True,
        "arms_bit_exact": True,
        "correctness": {
            "scenario_0_pre": correctness_scenario_0_pre,
            "scenario_0_post": correctness_scenario_0_post,
            "scenario_1_live_mutation": correctness_scenario_1,
        },
        "allocation_before_replay": allocation_before,
        "allocation_after_replay": allocation_after,
        "allocator_before": allocation_before,
        "allocator_after": allocation_after,
        "allocator_checks": allocator_checks,
        "allocator_stable": True,
        "zero_replay_allocations": True,
        "precondition": args.precondition,
        "precondition_seconds": args.precondition_seconds,
        "maximum_precondition_seconds": args.maximum_precondition_seconds,
        "max_sm_clock_delta_mhz": args.max_sm_clock_delta_mhz,
        "warmup": args.warmup,
        "cycles": args.cycles,
        "event_batch_cycles": args.event_batch_cycles,
        "conditions": conditions,
    }


def main() -> None:
    args = _args()
    result = _run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "case": result["case"],
                "correctness": result["correctness"],
                "ratios_b_over_a": {
                    condition: payload["timings"]["ratios_b_over_a"]
                    for condition, payload in result["conditions"].items()
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
