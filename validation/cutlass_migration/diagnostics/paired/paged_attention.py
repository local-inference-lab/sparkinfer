#!/usr/bin/env python3
"""Exact-object same-address CUDA-graph ABBA for SM120 paged attention.

The cases are the production migration-corpus routes: six FP8-KV prefill
sizes, BF16 direct decode, BF16 direct dual-TMA prefill, and BF16/FP8 split
verify including the persistent merge.  The current production planner and
launcher construct every runtime argument, but their launch hook is pinned to
immutable cache objects so this script cannot silently compile or time a
different specialization.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
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
from validation.cutlass_migration.paths import REPO_ROOT
import b12x.attention.paged.api as paged_api
from b12x.cute.compiler import b12x_package_fingerprint
from b12x.attention.paged.reference import paged_attention_reference
from b12x.integration.attention import (
    clear_attention_caches,
    paged_attention_forward,
)
from tests.paged_attention_helpers import (
    make_paged_inputs,
    quantize_paged_kv_cache_e4m3,
)
from tests.test_cute_migration_paged_corpus import (
    _make_fixed_graph_binding,
)


_Q_HEADS = 8
_KV_HEADS = 1
_HEAD_DIM = 256
_PAGE_SIZE = 64
_MERGE_SPEC = "4507cc4084fd17d5ee8fc6a02b03fe28118c1e15ac67d956717693b9f8e997a4"

_REQUIRED_RUNTIME_SOURCES = frozenset(
    {
        "benchmarks/common.py",
        "validation/cutlass_migration/diagnostics/paired/paged_attention.py",
        "validation/cutlass_migration/core/exact_cache_abba.py",
        "b12x/attention/paged/api.py",
        "b12x/attention/paged/forward_extend_generic.py",
        "b12x/attention/paged/forward_paged.py",
        "b12x/attention/paged/merge.py",
        "b12x/attention/paged/planner.py",
        "b12x/attention/paged/reference.py",
        "b12x/attention/paged/traits.py",
        "b12x/attention/paged/workspace.py",
        "b12x/cute/compiler.py",
        "b12x/integration/attention.py",
        "b12x/integration/paged_attention_scratch.py",
        "tests/paged_attention_helpers.py",
        "tests/test_cute_migration_paged_corpus.py",
    }
)


@dataclass(frozen=True)
class PagedCase:
    name: str
    q_len: int
    cache_len: int
    mode: str
    fp8_kv: bool
    disable_split_kv: bool
    spec_hashes: tuple[str, ...]


_CASES = {
    case.name: case
    for case in (
        PagedCase(
            "prefill-q8-fp8",
            8,
            8,
            "extend",
            True,
            False,
            ("aef26a0ded3d4dbb9ba59dc61c4508ed2d1ab46f7347905efa6b0a2d9a21e8e0",),
        ),
        PagedCase(
            "prefill-q16-fp8",
            16,
            16,
            "extend",
            True,
            False,
            ("aef26a0ded3d4dbb9ba59dc61c4508ed2d1ab46f7347905efa6b0a2d9a21e8e0",),
        ),
        PagedCase(
            "prefill-q64-fp8",
            64,
            64,
            "extend",
            True,
            False,
            ("aef26a0ded3d4dbb9ba59dc61c4508ed2d1ab46f7347905efa6b0a2d9a21e8e0",),
        ),
        PagedCase(
            "prefill-q128-fp8",
            128,
            128,
            "extend",
            True,
            False,
            ("840a669c8327df474a5c91cba02c0c0b5c8bffc340546f170aaec9184d115dca",),
        ),
        PagedCase(
            "prefill-q256-fp8",
            256,
            256,
            "extend",
            True,
            False,
            ("970344de79a2930e432a8a92baa666062a398ac4a361fdf585c7f84ebd03e901",),
        ),
        PagedCase(
            "prefill-q1024-fp8",
            1024,
            1024,
            "extend",
            True,
            False,
            ("c186e261c0a52d69b78bbf1bfa21b9da681374943af3bdaa04802b6a557b3909",),
        ),
        PagedCase(
            "decode-q1-bf16-direct",
            1,
            256,
            "decode",
            False,
            True,
            ("2c9d0262329787d8a1efc769bed65962b93b92c405c90d5c9dc203d519e1df92",),
        ),
        PagedCase(
            "prefill-q4-bf16-direct-dual-tma-tail",
            4,
            4096,
            "extend",
            False,
            True,
            ("bd89ac7c3abdcac618cc04cdc7e46cac5ff27cf256dd49baf507a90497f22d8b",),
        ),
        PagedCase(
            "verify-q4-bf16-split",
            4,
            256,
            "verify",
            False,
            False,
            (
                "1ab8d5afe3a68c85a4fd9ed776c65667d5a9bf4a1fb92bd670d0dcc72674e6e6",
                _MERGE_SPEC,
            ),
        ),
        PagedCase(
            "verify-q4-fp8-split",
            4,
            256,
            "verify",
            True,
            False,
            (
                "83958ad87fb337d19c601b7d5d048f1deb957aec29be927a9ad5020bea8fc102",
                _MERGE_SPEC,
            ),
        ),
    )
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_evidence_status_argument(parser)
    parser.add_argument("--a-cache", type=Path, required=True)
    parser.add_argument("--a-label", default="cutlass-4.5.2")
    parser.add_argument("--b-cache", type=Path, required=True)
    parser.add_argument("--b-label", default="cutlass-4.6.0")
    parser.add_argument("--case", choices=tuple(_CASES), required=True)
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


def _source_path(module: object) -> Path | None:
    raw_path = getattr(module, "__file__", None)
    if not isinstance(raw_path, str):
        return None
    path = Path(raw_path)
    if path.suffix in {".pyc", ".pyo"}:
        try:
            path = Path(importlib.util.source_from_cache(str(path)))
        except ValueError:
            return None
    try:
        path = path.resolve()
        relative = path.relative_to(REPO_ROOT)
    except (OSError, ValueError):
        return None
    if not relative.parts or relative.parts[0] not in {"b12x", "benchmarks", "tests"}:
        return None
    if path.suffix != ".py" or not path.is_file():
        return None
    return path


def _imported_runtime_source_sha256() -> dict[str, str]:
    sources: dict[str, str] = {}
    for module in tuple(sys.modules.values()):
        path = _source_path(module)
        if path is None:
            continue
        relative = path.relative_to(REPO_ROOT).as_posix()
        digest = sha256_file(path)
        previous = sources.setdefault(relative, digest)
        if previous != digest:
            raise RuntimeError(f"runtime source changed while hashing {relative}")
    missing = sorted(_REQUIRED_RUNTIME_SOURCES - set(sources))
    if missing:
        raise RuntimeError(
            f"paged ABBA did not import required runtime sources: {missing}"
        )
    return dict(sorted(sources.items()))


def _artifact_package_fingerprints(
    artifacts: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, str]:
    by_label: dict[str, str] = {}
    for label, records in artifacts.items():
        fingerprints = {
            str(record.get("package_fingerprint", "")) for record in records.values()
        }
        if len(fingerprints) != 1 or "" in fingerprints:
            raise RuntimeError(
                f"{label} exact objects do not have one package fingerprint: "
                f"{sorted(fingerprints)}"
            )
        by_label[label] = next(iter(fingerprints))
    return by_label


def _require_matching_artifact_package_fingerprints(
    artifacts: dict[str, dict[str, dict[str, Any]]],
    runtime_package_fingerprint: str,
) -> dict[str, str]:
    artifact_package_fingerprints = _artifact_package_fingerprints(artifacts)
    mismatched = {
        label: fingerprint
        for label, fingerprint in artifact_package_fingerprints.items()
        if fingerprint != runtime_package_fingerprint
    }
    if mismatched:
        raise RuntimeError(
            "current b12x package fingerprint differs from exact-cache frozen "
            f"source: runtime={runtime_package_fingerprint}, "
            f"artifacts={artifact_package_fingerprints}"
        )
    return artifact_package_fingerprints


def _tensor_pointers(tensors: dict[str, torch.Tensor]) -> dict[str, int]:
    return {name: int(tensor.data_ptr()) for name, tensor in tensors.items()}


def _assert_hashes(
    tensors: dict[str, torch.Tensor],
    expected: dict[str, str],
    *,
    description: str,
) -> dict[str, str]:
    observed = {name: tensor_sha256(tensor) for name, tensor in tensors.items()}
    if observed != expected:
        changed = {
            name: {"expected": expected.get(name), "observed": digest}
            for name, digest in observed.items()
            if expected.get(name) != digest
        }
        raise AssertionError(f"{description} tensor hashes changed: {changed}")
    return observed


def _correctness_metrics(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> dict[str, object]:
    actual_f32 = actual.float()
    expected_f32 = expected.float()
    difference = actual_f32 - expected_f32
    denominator = float(torch.linalg.vector_norm(expected_f32).item())
    relative_l2 = float(torch.linalg.vector_norm(difference).item()) / max(
        denominator,
        1e-12,
    )
    cosine = float(
        torch.nn.functional.cosine_similarity(
            actual_f32.reshape(-1),
            expected_f32.reshape(-1),
            dim=0,
        ).item()
    )
    return {
        "finite": bool(torch.isfinite(actual_f32).all().item()),
        "nonzero": int(torch.count_nonzero(actual_f32).item()),
        "max_abs": float(difference.abs().max().item()),
        "relative_l2": relative_l2,
        "cosine": cosine,
        "sha256": tensor_sha256(actual),
    }


def _assert_correct(
    *,
    label: str,
    output: torch.Tensor,
    lse_base2: torch.Tensor,
    expected: torch.Tensor,
    expected_lse: torch.Tensor,
    fp8_kv: bool,
) -> dict[str, object]:
    output_metrics = _correctness_metrics(output, expected)
    lse = lse_base2.float() * math.log(2.0)
    lse_metrics = _correctness_metrics(lse, expected_lse)
    minimum_cosine = 0.999 if fp8_kv else 0.99999
    maximum_abs = 0.06 if fp8_kv else 0.03
    maximum_lse_abs = 0.09 if fp8_kv else 0.05
    if (
        not output_metrics["finite"]
        or int(output_metrics["nonzero"]) == 0
        or float(output_metrics["cosine"]) < minimum_cosine
        or float(output_metrics["max_abs"]) > maximum_abs
        or not lse_metrics["finite"]
        or float(lse_metrics["max_abs"]) > maximum_lse_abs
    ):
        raise AssertionError(
            f"{label} failed the paged Torch oracle: "
            f"output={output_metrics}, lse={lse_metrics}"
        )
    return {
        "output": output_metrics,
        "lse": lse_metrics,
        "thresholds": {
            "minimum_cosine": minimum_cosine,
            "maximum_abs": maximum_abs,
            "maximum_lse_abs": maximum_lse_abs,
        },
    }


def _prepare_live_metadata_scenario(
    *,
    case: PagedCase,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
) -> dict[str, object]:
    if case.cache_len <= 1:
        raise RuntimeError("paged live metadata proof requires cache_len > 1")
    canonical_page_count = (case.cache_len + _PAGE_SIZE - 1) // _PAGE_SIZE
    canonical_page_ids = [
        int(value)
        for value in page_table[0, :canonical_page_count].detach().cpu().tolist()
    ]
    if not canonical_page_ids:
        raise RuntimeError("paged live metadata proof has no canonical cache pages")
    used_page_ids = set(canonical_page_ids)
    alternate_page_id = next(
        (
            page_id
            for page_id in range(int(k_cache.shape[0]))
            if page_id not in used_page_ids
        ),
        None,
    )
    if alternate_page_id is None:
        raise RuntimeError("paged live metadata proof requires one unused cache page")

    source_page_id = canonical_page_ids[0]
    k_cache[alternate_page_id].copy_(k_cache[source_page_id])
    v_cache[alternate_page_id].copy_(
        (-v_cache[source_page_id].float()).to(v_cache.dtype)
    )

    live_page_table = page_table.clone()
    live_page_table[0, 0] = int(alternate_page_id)
    live_cache_seqlens = cache_seqlens.clone()
    live_cache_seqlens[0] = int(case.cache_len - 1)
    live_cu_seqlens_q = cu_seqlens_q.clone()
    live_q_len = case.q_len
    if case.q_len == case.cache_len:
        # Prefill is right-aligned causal attention. Shorten Q with K so the
        # second scenario remains a valid serving request instead of creating
        # an all-masked first row.
        live_q_len -= 1
        live_cu_seqlens_q[-1] = int(live_q_len)
    if live_q_len <= 0 or live_q_len > case.cache_len - 1:
        raise RuntimeError(
            "paged live metadata scenario is not a valid right-aligned request: "
            f"q={live_q_len}, cache={case.cache_len - 1}"
        )
    return {
        "page_table": live_page_table,
        "cache_seqlens": live_cache_seqlens,
        "cu_seqlens_q": live_cu_seqlens_q,
        "live_q_len": live_q_len,
        "source_page_id": source_page_id,
        "alternate_page_id": alternate_page_id,
    }


def _capture_arm(
    *,
    label: str,
    compiled_by_spec: dict[str, object],
    expected_specs: tuple[str, ...],
    launch,
    stream: torch.cuda.Stream,
) -> tuple[torch.cuda.CUDAGraph, list[str]]:
    observed: list[str] = []
    with pin_module_launches(paged_api, compiled_by_spec, observed):
        with torch.cuda.stream(stream):
            launch()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph, stream=stream):
            launch()
        stream.synchronize()
    expected_set = set(expected_specs)
    if not observed or set(observed) != expected_set:
        raise RuntimeError(
            f"{label} production route observed specs {sorted(set(observed))}, "
            f"expected {sorted(expected_set)}"
        )
    if any(spec not in expected_set for spec in observed):
        raise RuntimeError(f"{label} observed an unpinned compile specification")
    return graph, observed


@torch.inference_mode()
def _run(args: argparse.Namespace) -> dict[str, object]:
    gpu = require_target_gpu()
    expected_physical_gpu = int(gpu["physical_index"])
    gpu_mode_initial = gpu_mode_snapshot(expected_physical_gpu)
    case = _CASES[args.case]
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
    if args.cycles < 500 or args.cycles % 2 != 0:
        raise ValueError("--cycles must be an even integer of at least 500")
    if not args.cold_l2:
        raise ValueError("release ABBA evidence requires warm- and cold-L2 timing")

    caches = {labels[0]: args.a_cache.resolve(), labels[1]: args.b_cache.resolve()}
    compiled: dict[str, dict[str, object]] = {label: {} for label in labels}
    artifacts: dict[str, dict[str, dict[str, Any]]] = {label: {} for label in labels}
    for label in labels:
        for spec_hash in case.spec_hashes:
            exact, provenance = load_exact(caches[label], spec_hash)
            compiled[label][spec_hash] = exact
            artifacts[label][spec_hash] = provenance
    runtime_package_fingerprint = b12x_package_fingerprint()
    artifact_package_fingerprints = _require_matching_artifact_package_fingerprints(
        artifacts,
        runtime_package_fingerprint,
    )
    artifact_verification_before = {
        label: {
            spec_hash: verify_artifact(provenance)
            for spec_hash, provenance in artifacts[label].items()
        }
        for label in labels
    }
    for spec_hash in case.spec_hashes:
        a = artifacts[labels[0]][spec_hash]
        b = artifacts[labels[1]][spec_hash]
        if a["compile_spec_json"] != b["compile_spec_json"]:
            raise RuntimeError(f"A/B compile specifications differ for {spec_hash}")
        if a["kernel_id"] != b["kernel_id"]:
            raise RuntimeError(f"A/B kernel ids differ for {spec_hash}")

    clear_attention_caches()
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = make_paged_inputs(
        q_seqlens=[case.q_len],
        cache_seqlens=[case.cache_len],
        page_size=_PAGE_SIZE,
        q_heads=_Q_HEADS,
        kv_heads=_KV_HEADS,
        head_dim=_HEAD_DIM,
        dtype=torch.bfloat16,
        seed=20260719 + case.q_len + 17 * case.cache_len,
    )
    k_descale = None
    v_descale = None
    if case.fp8_kv:
        k_cache, v_cache, k_descale, v_descale = quantize_paged_kv_cache_e4m3(
            k_cache,
            v_cache,
            page_table,
            cache_seqlens,
        )
        k_descale = k_descale.reshape(-1)
        v_descale = v_descale.reshape(-1)
    live_scenario = _prepare_live_metadata_scenario(
        case=case,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
    )
    expected, expected_lse = paged_attention_reference(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        k_descale=k_descale,
        v_descale=v_descale,
        causal=True,
    )
    live_q_len = int(live_scenario["live_q_len"])
    live_expected, live_expected_lse = paged_attention_reference(
        q[:live_q_len],
        k_cache,
        v_cache,
        live_scenario["page_table"],
        live_scenario["cache_seqlens"],
        live_scenario["cu_seqlens_q"],
        k_descale=k_descale,
        v_descale=v_descale,
        causal=True,
    )
    output = torch.empty_like(q)
    binding, scratch, scratch_plan = _make_fixed_graph_binding(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        output=output,
        mode=case.mode,
        disable_split_kv=case.disable_split_kv,
        k_descale=k_descale,
        v_descale=v_descale,
    )
    live_inputs = {
        "page_table": binding.scratch.page_table,
        "cache_seqlens": binding.scratch.cache_seqlens,
        "cu_seqlens_q": binding.scratch.cu_seqlens_q,
    }
    if any(tensor is None for tensor in live_inputs.values()):
        raise RuntimeError("paged graph binding did not materialize live metadata")
    live_inputs = {
        name: tensor for name, tensor in live_inputs.items() if tensor is not None
    }
    source_metadata = {
        "source_page_table": page_table,
        "source_cache_seqlens": cache_seqlens,
        "source_cu_seqlens_q": cu_seqlens_q,
    }
    canonical_live_hashes = {
        name: tensor_sha256(tensor) for name, tensor in live_inputs.items()
    }
    source_metadata_hashes = {
        name.removeprefix("source_"): tensor_sha256(tensor)
        for name, tensor in source_metadata.items()
    }
    if canonical_live_hashes != source_metadata_hashes:
        raise RuntimeError(
            "paged graph scratch metadata does not match the captured source values: "
            f"scratch={canonical_live_hashes}, source={source_metadata_hashes}"
        )
    canonical_live_values = {
        name: tensor.clone() for name, tensor in live_inputs.items()
    }
    if bool(binding.scratch.plan.split_kv) != (_MERGE_SPEC in case.spec_hashes):
        raise RuntimeError(
            f"planner split_kv={binding.scratch.plan.split_kv} violates {case.name} contract"
        )

    immutable_inputs = {
        "q": q,
        "k_cache": k_cache,
        "v_cache": v_cache,
        **source_metadata,
        **({"k_descale": k_descale} if k_descale is not None else {}),
        **({"v_descale": v_descale} if v_descale is not None else {}),
    }
    immutable_input_hashes = {
        name: tensor_sha256(tensor) for name, tensor in immutable_inputs.items()
    }
    stable_tensors = {
        **immutable_inputs,
        **{f"live_{name}": tensor for name, tensor in live_inputs.items()},
        "output": output,
        "lse": binding.scratch.lse,
        **{f"scratch_{index}": tensor for index, tensor in enumerate(scratch)},
    }
    if any(tensor is None for tensor in stable_tensors.values()):
        raise RuntimeError("paged graph binding has an unmaterialized stable tensor")
    stable_tensors = {
        name: tensor for name, tensor in stable_tensors.items() if tensor is not None
    }
    fixed_pointers = _tensor_pointers(stable_tensors)

    def launch() -> tuple[torch.Tensor, torch.Tensor]:
        return paged_attention_forward(binding=binding)

    stream = torch.cuda.Stream()
    graphs: dict[str, torch.cuda.CUDAGraph] = {}
    observed_specs: dict[str, list[str]] = {}
    for label in labels:
        graph, observed = _capture_arm(
            label=label,
            compiled_by_spec=compiled[label],
            expected_specs=case.spec_hashes,
            launch=launch,
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
    expected_kernel_nodes = len(case.spec_hashes)
    if any(
        topology["kernel_node_count"] != expected_kernel_nodes
        for topology in topologies.values()
    ):
        raise AssertionError(
            f"expected {expected_kernel_nodes} kernel graph nodes, got {topologies}"
        )

    allocator_checks: list[dict[str, object]] = []
    correctness: dict[str, dict[str, object]] = {}
    arm_outputs: dict[str, torch.Tensor] = {}
    arm_lse: dict[str, torch.Tensor] = {}
    for label in labels:
        with torch.cuda.stream(stream):
            output.fill_(float("nan"))
            binding.scratch.lse.fill_(float("nan"))
        stream.synchronize()
        counters_before = allocator_counters()
        with torch.cuda.stream(stream):
            graphs[label].replay()
        stream.synchronize()
        counters_after = allocator_counters()
        if counters_before != counters_after:
            raise AssertionError(
                f"CUDA allocator state changed during scenario-0 {label} replay"
            )
        allocator_checks.append(
            {
                "scenario": "scenario_0_pre_timing",
                "label": label,
                "before": counters_before,
                "after": counters_after,
                "unchanged": True,
            }
        )
        lse = binding.scratch.current_lse_view()
        correctness[label] = _assert_correct(
            label=label,
            output=output,
            lse_base2=lse,
            expected=expected,
            expected_lse=expected_lse,
            fp8_kv=case.fp8_kv,
        )
        arm_outputs[label] = output.clone()
        arm_lse[label] = lse.clone()
    if not torch.equal(arm_outputs[labels[0]], arm_outputs[labels[1]]):
        raise AssertionError("A/B outputs are not bit exact")
    if not torch.equal(arm_lse[labels[0]], arm_lse[labels[1]]):
        raise AssertionError("A/B LSE outputs are not bit exact")

    live_output_buffers = {label: torch.empty_like(output) for label in labels}
    live_lse_buffers = {
        label: torch.empty_like(binding.scratch.current_lse_view()) for label in labels
    }
    scenario_inputs = {
        name: live_scenario[name]
        for name in ("page_table", "cache_seqlens", "cu_seqlens_q")
    }
    with torch.cuda.stream(stream):
        for name, tensor in live_inputs.items():
            tensor.copy_(scenario_inputs[name])
    stream.synchronize()
    live_input_hashes = {
        name: tensor_sha256(tensor) for name, tensor in live_inputs.items()
    }
    input_change_by_field = {
        name: live_input_hashes[name] != canonical_live_hashes[name]
        for name in live_inputs
    }
    changed_inputs = {
        name: changed for name, changed in input_change_by_field.items() if changed
    }
    if (
        not input_change_by_field["page_table"]
        or not input_change_by_field["cache_seqlens"]
    ):
        raise AssertionError(
            "paged live scenario did not change page_table and cache_seqlens: "
            f"{input_change_by_field}"
        )
    if live_q_len != case.q_len and not input_change_by_field["cu_seqlens_q"]:
        raise AssertionError("paged live prefill scenario did not shorten cu_seqlens_q")
    if _tensor_pointers(stable_tensors) != fixed_pointers:
        raise AssertionError(
            "paged tensor addresses changed during live-input mutation"
        )
    _assert_hashes(
        immutable_inputs,
        immutable_input_hashes,
        description="paged immutable inputs before live replay",
    )

    allocation_before_live_replay = allocator_counters()
    for label in labels:
        with torch.cuda.stream(stream):
            output.fill_(float("nan"))
            binding.scratch.lse.fill_(float("nan"))
        stream.synchronize()
        counters_before = allocator_counters()
        with torch.cuda.stream(stream):
            graphs[label].replay()
            live_output_buffers[label].copy_(output)
            live_lse_buffers[label].copy_(binding.scratch.current_lse_view())
        stream.synchronize()
        counters_after = allocator_counters()
        if counters_before != counters_after:
            raise AssertionError(
                f"CUDA allocator state changed during scenario-1 {label} replay"
            )
        allocator_checks.append(
            {
                "scenario": "scenario_1_live_mutation",
                "label": label,
                "before": counters_before,
                "after": counters_after,
                "unchanged": True,
            }
        )
    allocation_after_live_replay = allocator_counters()
    if allocation_before_live_replay != allocation_after_live_replay:
        raise AssertionError(
            "CUDA allocator state changed across paged live-input graph replays: "
            f"before={allocation_before_live_replay}, "
            f"after={allocation_after_live_replay}"
        )

    live_correctness: dict[str, dict[str, object]] = {}
    output_change_by_arm: dict[str, dict[str, object]] = {}
    for label in labels:
        live_output = live_output_buffers[label][:live_q_len]
        live_lse = live_lse_buffers[label][:live_q_len]
        live_correctness[label] = _assert_correct(
            label=f"{label} live metadata",
            output=live_output,
            lse_base2=live_lse,
            expected=live_expected,
            expected_lse=live_expected_lse,
            fp8_kv=case.fp8_kv,
        )
        canonical_output = arm_outputs[label][:live_q_len]
        changed = not torch.equal(canonical_output, live_output)
        output_change_by_arm[label] = {
            "changed": changed,
            "canonical_sha256": tensor_sha256(canonical_output),
            "live_sha256": tensor_sha256(live_output),
        }
        if not changed:
            raise AssertionError(
                f"{label} captured paged graph ignored live metadata changes"
            )
    if not torch.equal(
        live_output_buffers[labels[0]][:live_q_len],
        live_output_buffers[labels[1]][:live_q_len],
    ):
        raise AssertionError("live-metadata A/B outputs are not bit exact")
    if not torch.equal(
        live_lse_buffers[labels[0]][:live_q_len],
        live_lse_buffers[labels[1]][:live_q_len],
    ):
        raise AssertionError("live-metadata A/B LSE outputs are not bit exact")
    changed_outputs = {
        label: bool(payload["changed"])
        for label, payload in output_change_by_arm.items()
    }
    scenario_0_output_sha256 = {
        label: tensor_sha256(arm_outputs[label][:live_q_len]) for label in labels
    }
    scenario_1_output_sha256 = {
        label: tensor_sha256(live_output_buffers[label][:live_q_len])
        for label in labels
    }

    with torch.cuda.stream(stream):
        for name, tensor in live_inputs.items():
            tensor.copy_(canonical_live_values[name])
    stream.synchronize()
    restored_live_hashes = _assert_hashes(
        live_inputs,
        canonical_live_hashes,
        description="paged restored live inputs",
    )
    _assert_hashes(
        immutable_inputs,
        immutable_input_hashes,
        description="paged immutable inputs after live replay",
    )
    if _tensor_pointers(stable_tensors) != fixed_pointers:
        raise AssertionError("paged tensor addresses changed after live-input restore")
    restored_correctness: dict[str, dict[str, object]] = {}
    for label in labels:
        with torch.cuda.stream(stream):
            output.fill_(float("nan"))
            binding.scratch.lse.fill_(float("nan"))
        stream.synchronize()
        counters_before = allocator_counters()
        with torch.cuda.stream(stream):
            graphs[label].replay()
        stream.synchronize()
        counters_after = allocator_counters()
        if counters_before != counters_after:
            raise AssertionError(
                f"CUDA allocator state changed during restored {label} replay"
            )
        allocator_checks.append(
            {
                "scenario": "scenario_0_restored",
                "label": label,
                "before": counters_before,
                "after": counters_after,
                "unchanged": True,
            }
        )
        restored_correctness[label] = _assert_correct(
            label=f"{label} restored metadata",
            output=output,
            lse_base2=binding.scratch.current_lse_view(),
            expected=expected,
            expected_lse=expected_lse,
            fp8_kv=case.fp8_kv,
        )

    # Allocate the fixed cold-L2 sweep before the replay-allocation baseline.
    if args.cold_l2:
        make_l2_flush_fn(True, args.l2_flush_bytes)
    allocation_before_replay = allocator_counters()
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
    condition_spec_hashes = list(case.spec_hashes)
    for condition in conditions.values():
        condition["compile_spec_hashes"] = condition_spec_hashes
        condition["all_graph_spec_hashes"] = condition_spec_hashes
    allocation_after_replay = allocator_counters()
    if allocation_before_replay != allocation_after_replay:
        raise AssertionError(
            "CUDA allocator state changed across graph replay benchmark: "
            f"before={allocation_before_replay}, after={allocation_after_replay}"
        )
    allocator_checks.append(
        {
            "scenario": "timed_scenario_0",
            "label": "all_arms",
            "before": allocation_before_replay,
            "after": allocation_after_replay,
            "unchanged": True,
        }
    )

    immutable_input_hashes_after = _assert_hashes(
        immutable_inputs,
        immutable_input_hashes,
        description="paged immutable inputs after timing",
    )
    timing_live_hashes = _assert_hashes(
        live_inputs,
        canonical_live_hashes,
        description="paged canonical live inputs after timing",
    )
    if _tensor_pointers(stable_tensors) != fixed_pointers:
        raise AssertionError("paged tensor addresses changed during timing")
    post_timing_correctness: dict[str, dict[str, object]] = {}
    for label in labels:
        with torch.cuda.stream(stream):
            output.fill_(float("nan"))
            binding.scratch.lse.fill_(float("nan"))
        stream.synchronize()
        counters_before = allocator_counters()
        with torch.cuda.stream(stream):
            graphs[label].replay()
        stream.synchronize()
        counters_after = allocator_counters()
        if counters_before != counters_after:
            raise AssertionError(
                f"CUDA allocator state changed during post-timing {label} replay"
            )
        allocator_checks.append(
            {
                "scenario": "scenario_0_post_timing",
                "label": label,
                "before": counters_before,
                "after": counters_after,
                "unchanged": True,
            }
        )
        post_timing_correctness[label] = _assert_correct(
            label=f"{label} post-timing",
            output=output,
            lse_base2=binding.scratch.current_lse_view(),
            expected=expected,
            expected_lse=expected_lse,
            fp8_kv=case.fp8_kv,
        )
    artifact_verification_after = {
        label: {
            spec_hash: verify_artifact(provenance)
            for spec_hash, provenance in artifacts[label].items()
        }
        for label in labels
    }
    if artifact_verification_after != artifact_verification_before:
        raise RuntimeError("exact cache artifacts changed during benchmark")
    if b12x_package_fingerprint() != runtime_package_fingerprint:
        raise RuntimeError("b12x package fingerprint changed during paged ABBA")
    gpu_mode_final = gpu_mode_snapshot(expected_physical_gpu)
    runtime_source_sha256 = _imported_runtime_source_sha256()
    return {
        "schema": "b12x.attention.paged.exact_cache_abba.v1",
        "evidence_status": args.evidence_status,
        "case": {
            "name": case.name,
            "q_len": case.q_len,
            "cache_len": case.cache_len,
            "mode": case.mode,
            "fp8_kv": case.fp8_kv,
            "disable_split_kv": case.disable_split_kv,
            "split_kv": bool(binding.scratch.plan.split_kv),
            "spec_hashes": case.spec_hashes,
            "planner": str(binding.scratch.plan),
            "scratch_plan": str(scratch_plan),
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
            "imported_runtime_source_sha256": runtime_source_sha256,
            "required_runtime_sources": sorted(_REQUIRED_RUNTIME_SOURCES),
            "runtime_sources_complete": True,
            "runtime_b12x_package_fingerprint": runtime_package_fingerprint,
            "artifact_b12x_package_fingerprints": artifact_package_fingerprints,
            "packages": _package_versions(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
        "runtime_b12x_package_fingerprint": runtime_package_fingerprint,
        "artifacts": artifacts,
        "artifact_verification_before": artifact_verification_before,
        "artifact_verification_after": artifact_verification_after,
        "observed_compile_specs": observed_specs,
        "production_planner": True,
        "production_launcher": True,
        "exact_cache_objects": True,
        "no_recompile": True,
        "cuda_graph_replay": True,
        "cuda_graph_topology": topologies,
        "cuda_graph_topology_equal": topology_equal,
        "fixed_workspace_capacity": True,
        "fixed_allocation": True,
        "same_address_arms": True,
        "fixed_pointers": fixed_pointers,
        "input_hashes": {
            "immutable": immutable_input_hashes,
            "live_scenario_0": canonical_live_hashes,
        },
        "read_only_inputs_immutable": True,
        "read_only_inputs": {
            "sha256_before": immutable_input_hashes,
            "sha256_after": immutable_input_hashes_after,
            "unchanged": immutable_input_hashes_after == immutable_input_hashes,
            "timed_live_scenario_0": {
                "unchanged": timing_live_hashes == canonical_live_hashes,
                "sha256_before": canonical_live_hashes,
                "sha256_after": timing_live_hashes,
            },
        },
        "poisoned_outputs_overwritten": True,
        "arms_bit_exact": True,
        "correctness": correctness,
        "restored_correctness": restored_correctness,
        "post_timing_correctness": post_timing_correctness,
        "live_input_scenarios_distinct": True,
        "live_input_mutation_changed_input": True,
        "live_input_mutation_changed_output": True,
        "live_input_mutation": {
            "captured_graph_replay": True,
            "captured_graph_reused": True,
            "in_place": True,
            "changed_input": True,
            "changed_output": True,
            "scenarios_distinct": True,
            "mutated_inputs": [
                name for name, changed in changed_inputs.items() if changed
            ],
            "changed_inputs": changed_inputs,
            "input_change_by_field": input_change_by_field,
            "changed_outputs": changed_outputs,
            "scenario_0_sha256": {
                name: canonical_live_hashes[name] for name in changed_inputs
            },
            "scenario_1_sha256": {
                name: live_input_hashes[name] for name in changed_inputs
            },
            "scenario_0_output_sha256": scenario_0_output_sha256,
            "scenario_1_output_sha256": scenario_1_output_sha256,
            "scenario_0": {
                "q_len": case.q_len,
                "cache_len": case.cache_len,
                "input_sha256": canonical_live_hashes,
            },
            "scenario_1": {
                "q_len": live_q_len,
                "cache_len": case.cache_len - 1,
                "source_page_id": live_scenario["source_page_id"],
                "alternate_page_id": live_scenario["alternate_page_id"],
                "input_sha256": live_input_hashes,
            },
            "restored_input_sha256": restored_live_hashes,
            "post_timing_input_sha256": timing_live_hashes,
            "output_change_by_arm": output_change_by_arm,
            "correctness": live_correctness,
            "arms_bit_exact": True,
            "lse_arms_bit_exact": True,
            "poisoned_checked_regions_overwritten": True,
            "same_addresses": True,
            "allocation_addresses_stable": True,
            "fixed_pointers": fixed_pointers,
            "allocator_before": allocation_before_live_replay,
            "allocator_after": allocation_after_live_replay,
            "allocator_stable": (
                allocation_before_live_replay == allocation_after_live_replay
            ),
            "read_only_inputs_unchanged": True,
        },
        "allocation_before_replay": allocation_before_replay,
        "allocation_after_replay": allocation_after_replay,
        "allocator_stable": True,
        "zero_replay_allocations": True,
        "allocator_checks": allocator_checks,
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
