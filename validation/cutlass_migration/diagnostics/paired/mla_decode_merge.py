#!/usr/bin/env python3
"""Exact-object CUDA-graph ABBA for SM120 MLA decode and split merge.

The decode cases pair the frozen decode object with the frozen one-chunk merge
object used by the real production entrypoint.  The direct merge cases isolate
the five-chunk plain and sink-folding kernels.  Every case uses caller-owned,
fixed-capacity storage, captures both arms at identical addresses, checks a GPU
oracle and exact arm equality before and after timing, proves zero replay-time
allocator activity, and records immutable cache-object provenance.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Callable

import torch

from benchmarks.common import (
    make_l2_flush_fn,
    resolve_l2_flush_bytes,
)
from validation.cutlass_migration.core.evidence_status import (
    add_evidence_status_argument,
)
from validation.cutlass_migration.core.gpu_scope import (
    add_target_gpu_argument,
    require_target_gpu,
)
from validation.cutlass_migration.core.exact_cache_abba import (
    gpu_mode_snapshot,
    time_conditions,
)
from validation.cutlass_migration.diagnostics.paired.mla_prefill_mg import (
    _correctness_metrics,
    _git_output,
    _graph_topology,
    _json_sha256,
    _manifest_for_spec,
    _sha256_file,
    _tensor_sha256,
    _verify_artifact,
)
from validation.cutlass_migration.paths import DATA_ROOT, PACKAGE_ROOT, REPO_ROOT
import b12x.attention.mla.kernel as mla_kernel
import b12x.attention.mla.merge as mla_merge
import b12x.cute.compiler as cute_compiler
from b12x.attention.mla.compressed_reference import (
    COMPRESSED_MLA_HEAD_DIM,
)
from b12x.integration.compressed_scratch import (
    B12XCompressedMLAScratchCaps,
    _compressed_mla_scratch_layout,
    _materialize_compressed_mla_scratch,
)
from tests.test_attention_mla_merge import (
    _install_scenario as _install_merge_scenario,
    _make_fixed_merge_problem,
    _make_merge_scenarios,
    _split_merge_fp32_oracle,
)
from tests.test_attention_mla_unified_corpus import (
    _ALLOCATOR_COUNTERS,
    _GLM_Q_DIM,
    _GLM_SM_SCALE,
    _GLM_V_DIM,
    _PAGE_SIZE,
    _SM_SCALE,
    _allocator_counters,
    _assert_output,
    _glm_reference,
    _install_glm_scenario,
    _install_scenario,
    _make_glm_inputs,
    _make_inputs,
    _poison_inactive_topk_tails,
    _reference,
)


_DECODE_MERGE_SPEC_HASH = (
    "27844cab02ed922c483a620b933e04948bdef60f25de74d8adddb2342bb14804"
)
_GLM_H8_NATIVE_ENV = "B12X_MLA_SM120_GLM_H8_NATIVE"
_DEFAULT_DECODE_ROWS = (1, 2)
_DEFAULT_MERGE_ROWS = (2, 4, 32, 128)


@dataclass(frozen=True)
class DecodeCase:
    name: str
    decode_spec_hash: str
    family: str
    per_token: bool
    has_extra: bool
    merge_spec_hash: str = _DECODE_MERGE_SPEC_HASH


@dataclass(frozen=True)
class MergeCase:
    name: str
    spec_hash: str
    with_sink: bool
    heads: int = 32
    chunks: int = 5


Case = DecodeCase | MergeCase


_CASES: dict[str, Case] = {
    case.name: case
    for case in (
        DecodeCase(
            name="dsv4-extra-per-token",
            decode_spec_hash=(
                "fd5e3a3fafd5b0a6c7caea5bd226b65b43217fcf51bfab9bad795481959763b6"
            ),
            family="dsv4",
            per_token=True,
            has_extra=True,
        ),
        DecodeCase(
            name="glm",
            decode_spec_hash=(
                "4a08fd70cf2a72687d46528da3fac6e0d14ec2cf21ad69b19b66651bd06d4200"
            ),
            family="glm",
            per_token=False,
            has_extra=False,
        ),
        DecodeCase(
            name="glm-per-token",
            decode_spec_hash=(
                "b2bbb8e080e5ceea33d603c3fdc336266ba2fe1209ac683c9e74919125c4103d"
            ),
            family="glm",
            per_token=True,
            has_extra=False,
        ),
        MergeCase(
            name="merge",
            spec_hash=(
                "6607e86d8a5d62134855d9f8c33dc17c93e7b5ab77fca984749690567d75ed06"
            ),
            with_sink=False,
        ),
        MergeCase(
            name="sink-merge",
            spec_hash=(
                "7ee273381cf0f95886d3cf65694f16fcdcabf37cb01859af9b9db85c04df25bd"
            ),
            with_sink=True,
        ),
    )
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_evidence_status_argument(parser)
    add_target_gpu_argument(parser)
    parser.add_argument("--case", choices=tuple(_CASES), required=True)
    parser.add_argument("--a-cache", type=Path, required=True)
    parser.add_argument(
        "--a-key",
        action="append",
        default=[],
        metavar="SPEC_HASH=CACHE_KEY",
        help="optional per-spec cache-key disambiguator; repeat as needed",
    )
    parser.add_argument("--a-label", default="cutlass-4.5.2")
    parser.add_argument("--b-cache", type=Path, required=True)
    parser.add_argument(
        "--b-key",
        action="append",
        default=[],
        metavar="SPEC_HASH=CACHE_KEY",
        help="optional per-spec cache-key disambiguator; repeat as needed",
    )
    parser.add_argument("--b-label", default="cutlass-4.6.0")
    parser.add_argument(
        "--rows",
        action="append",
        default=[],
        metavar="N[,N...]",
        help="row counts; repeat or use comma-separated values",
    )
    parser.add_argument("--precondition-replays", type=int, default=1000)
    parser.add_argument("--precondition-seconds", type=float, default=5.0)
    parser.add_argument("--maximum-precondition-seconds", type=float, default=60.0)
    parser.add_argument("--max-sm-clock-delta-mhz", type=float, default=60.0)
    parser.add_argument("--warmup-cycles", type=int, default=100)
    parser.add_argument("--cycles", type=int, default=600)
    parser.add_argument("--event-batch-cycles", type=int, default=50)
    parser.add_argument("--replays-per-reported-sample", type=int, default=1)
    parser.add_argument("--l2-flush-bytes", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _key_overrides(raw_values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in raw_values:
        spec_hash, separator, cache_key = raw.partition("=")
        if (
            not separator
            or not re.fullmatch(r"[0-9a-f]{64}", spec_hash)
            or not re.fullmatch(r"[0-9a-f]{64}", cache_key)
        ):
            raise ValueError(
                f"invalid cache-key override {raw!r}; expected SPEC_HASH=CACHE_KEY"
            )
        if spec_hash in result and result[spec_hash] != cache_key:
            raise ValueError(f"conflicting cache keys for spec {spec_hash}")
        result[spec_hash] = cache_key
    return result


def _row_sweep(raw_values: list[str], case: Case) -> tuple[int, ...]:
    default = (
        _DEFAULT_DECODE_ROWS if isinstance(case, DecodeCase) else _DEFAULT_MERGE_ROWS
    )
    if not raw_values:
        return default
    rows: list[int] = []
    for raw in raw_values:
        for value in raw.split(","):
            try:
                row = int(value.strip())
            except ValueError as error:
                raise ValueError(f"invalid --rows value {value!r}") from error
            if row <= 0:
                raise ValueError(f"row counts must be positive, got {row}")
            if isinstance(case, MergeCase) and row < 2:
                raise ValueError("direct merge scenarios require at least two rows")
            if isinstance(case, DecodeCase) and row > 2:
                raise ValueError(
                    "the frozen decode exception objects use a two-row workspace "
                    "stride contract; exact-object replay supports rows 1 or 2"
                )
            if row not in rows:
                rows.append(row)
    if not rows:
        raise ValueError("--rows did not contain a row count")
    return tuple(rows)


def _spec_facts(manifest: dict[str, Any]) -> tuple[str, int, dict[str, object]]:
    raw = json.loads(str(manifest["compile_spec_json"]))
    facts_raw = raw.get("facts")
    if not isinstance(facts_raw, list):
        raise RuntimeError("compile spec facts are not a list")
    facts = {
        str(fact[0]): fact[1]
        for fact in facts_raw
        if isinstance(fact, list) and len(fact) == 2
    }
    return str(raw.get("kernel")), int(raw.get("version", -1)), facts


def _validate_decode_manifest(
    manifest: dict[str, Any],
    case: DecodeCase,
) -> None:
    kernel_id, version, facts = _spec_facts(manifest)
    if kernel_id != "attention.mla.sm120.decode" or version != 17:
        raise RuntimeError(f"unexpected decode compile spec: {kernel_id} v{version}")
    expected = {
        "model_type": 0 if case.family == "dsv4" else 1,
        "compute_mode": 0,
        "scale_format": 0 if case.family == "dsv4" else 1,
        "num_heads": 8,
        "hpb": 16,
        "valid_hpb": 8,
        "grid_h_blocks": 1,
        "head_block_offset": 0,
        "num_splits": 1,
        "chunks_per_split": 2,
        "page_block_size": 64,
        "topk_bucket": 64 if case.family == "dsv4" else 128,
        "has_extra": int(case.has_extra),
        "extra_topk_bucket": 64 if case.has_extra else 0,
        "per_token_len": int(case.per_token),
        "native_glm_h8": int(case.family == "glm"),
        "native_dsv4_h8": int(case.family == "dsv4"),
        "native_dsv4_h16": 0,
    }
    mismatches = {
        name: {"expected": value, "actual": facts.get(name)}
        for name, value in expected.items()
        if facts.get(name) != value
    }
    if mismatches:
        raise RuntimeError(
            f"decode compile spec does not match {case.name}: {mismatches}"
        )


def _validate_merge_manifest(
    manifest: dict[str, Any],
    *,
    spec_hash: str,
    static_num_chunks: int,
    with_sink: bool,
) -> None:
    kernel_id, version, facts = _spec_facts(manifest)
    expected_kernel = "attention.mla.sink_merge" if with_sink else "attention.mla.merge"
    if kernel_id != expected_kernel or version != 4:
        raise RuntimeError(f"unexpected merge compile spec: {kernel_id} v{version}")
    expected = {"static_num_chunks": static_num_chunks}
    if with_sink:
        expected["kind"] = "attn_sink"
    mismatches = {
        name: {"expected": value, "actual": facts.get(name)}
        for name, value in expected.items()
        if facts.get(name) != value
    }
    if mismatches:
        raise RuntimeError(
            f"merge compile spec {spec_hash} violates its case contract: {mismatches}"
        )


def _load_exact(
    cache: Path,
    spec_hash: str,
    key: str | None,
    validate: Callable[[dict[str, Any]], None],
) -> tuple[Any, dict[str, Any]]:
    cache = cache.resolve()
    manifest_path, manifest = _manifest_for_spec(cache, spec_hash, key)
    validate(manifest)
    cache_key = str(manifest["cache_key"])
    object_path = manifest_path.with_suffix(".o")
    object_sha256 = _sha256_file(object_path)
    if object_sha256 != manifest["object_sha256"]:
        raise RuntimeError(f"object digest mismatch: {object_path}")

    previous_cache = os.environ.get("B12X_CUTE_COMPILE_CACHE_DIR")
    os.environ["B12X_CUTE_COMPILE_CACHE_DIR"] = str(cache)
    try:
        compiled = cute_compiler._load_cute_compile_from_disk(cache_key)
    finally:
        if previous_cache is None:
            os.environ.pop("B12X_CUTE_COMPILE_CACHE_DIR", None)
        else:
            os.environ["B12X_CUTE_COMPILE_CACHE_DIR"] = previous_cache
    if compiled is None:
        raise RuntimeError(f"failed to load exact cached object {object_path}")
    return compiled, {
        "cache": str(cache),
        "cache_key": cache_key,
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "object_path": str(object_path),
        "object_sha256": object_sha256,
        "object_bytes": object_path.stat().st_size,
        "compile_spec_hash": manifest["compile_spec_hash"],
        "compile_spec_json": manifest["compile_spec_json"],
        "semantic_key": manifest.get("semantic_key"),
        "package_fingerprint": manifest.get("package_fingerprint"),
        "target": manifest.get("target"),
        "kernel_id": manifest.get("kernel_id"),
        "compile_options": manifest.get("compile_options"),
        "compile_environment": manifest.get("compile_environment"),
        "toolchain": manifest.get("toolchain"),
    }


def _make_decode_workspace(
    *,
    rows: int,
    heads: int,
    width: int,
    family: str,
    device: torch.device,
):
    caps = B12XCompressedMLAScratchCaps(
        device=device,
        num_q_heads=heads,
        max_q_rows=rows,
        max_width=width,
        head_dim=_GLM_Q_DIM if family == "glm" else COMPRESSED_MLA_HEAD_DIM,
        v_head_dim=_GLM_V_DIM,
        max_chunks_per_row=8,
        page_size=_PAGE_SIZE,
    )
    layout = _compressed_mla_scratch_layout(caps)
    storage = torch.zeros(layout.nbytes, dtype=torch.uint8, device=device)
    return _materialize_compressed_mla_scratch(caps, storage, layout)


def _stable_pointer_check(
    tensors: dict[str, torch.Tensor],
    expected: dict[str, int],
) -> None:
    observed = {name: tensor.data_ptr() for name, tensor in tensors.items()}
    if observed != expected:
        raise AssertionError(f"stable tensor pointer changed: {expected} -> {observed}")


def _run_timing(
    *,
    graphs: dict[str, torch.cuda.CUDAGraph],
    labels: tuple[str, str],
    all_graph_spec_hashes: tuple[str, ...],
    precondition_replays: int,
    precondition_seconds: float,
    maximum_precondition_seconds: float,
    warmup_cycles: int,
    cycles: int,
    event_batch_cycles: int,
    replays_per_reported_sample: int,
    l2_flush_bytes: int,
    expected_physical_gpu: int,
    max_sm_clock_delta_mhz: float,
    stream: torch.cuda.Stream,
    device: torch.device,
) -> dict[str, object]:
    # Allocate the stable cold-L2 sweep before the enclosing replay-allocation
    # baseline. The shared timer has an additional per-condition allocation
    # invariant around the exact sample schedule.
    make_l2_flush_fn(True, l2_flush_bytes)
    allocator_before_timing = _allocator_counters(device)
    conditions = time_conditions(
        graphs,
        labels=labels,
        precondition=precondition_replays,
        warmup=warmup_cycles,
        cycles=cycles,
        event_batch_cycles=event_batch_cycles,
        stream=stream,
        cold_l2=True,
        l2_flush_bytes=l2_flush_bytes,
        replays_per_reported_sample=replays_per_reported_sample,
        precondition_seconds=precondition_seconds,
        maximum_precondition_seconds=maximum_precondition_seconds,
        mode_snapshot=lambda: gpu_mode_snapshot(expected_physical_gpu),
        required_pstate="P1",
        max_sm_clock_delta_mhz=max_sm_clock_delta_mhz,
    )
    allocator_after_timing = _allocator_counters(device)
    if allocator_after_timing != allocator_before_timing:
        raise AssertionError(
            "CUDA allocator state changed across MLA decode/merge graph timing: "
            f"{allocator_before_timing} -> {allocator_after_timing}"
        )
    compile_spec_hashes = {label: list(all_graph_spec_hashes) for label in labels}
    for condition in conditions.values():
        condition["compile_spec_hashes"] = compile_spec_hashes
        condition["all_graph_spec_hashes"] = list(all_graph_spec_hashes)
    return {
        "unit": "us",
        "precondition_replays": precondition_replays,
        "precondition_seconds": precondition_seconds,
        "maximum_precondition_seconds": maximum_precondition_seconds,
        "warmup_cycles_per_condition": warmup_cycles,
        "cycles": cycles,
        "replays_per_cycle": 4,
        "samples_per_arm_per_condition": cycles * 2,
        "event_batch_cycles": event_batch_cycles,
        "replays_per_reported_sample": replays_per_reported_sample,
        "max_sm_clock_delta_mhz": max_sm_clock_delta_mhz,
        "balanced_order": "alternating ABBA/BAAB",
        "allocator_before": allocator_before_timing,
        "allocator_after": allocator_after_timing,
        "allocator_stable": True,
        "conditions": conditions,
    }


def _run_decode_rows(
    *,
    rows: int,
    case: DecodeCase,
    labels: tuple[str, str],
    select_arm: Callable[[str], None],
    dispatch_counts: dict[str, dict[str, int]],
    precondition_replays: int,
    precondition_seconds: float,
    maximum_precondition_seconds: float,
    warmup_cycles: int,
    cycles: int,
    event_batch_cycles: int,
    replays_per_reported_sample: int,
    l2_flush_bytes: int,
    expected_physical_gpu: int,
    max_sm_clock_delta_mhz: float,
    device: torch.device,
) -> dict[str, object]:
    heads = 8
    if case.family == "dsv4":
        main_width = 64
        extra_width = 64 if case.has_extra else 0
        inputs = _make_inputs(
            rows=rows,
            heads=heads,
            main_width=main_width,
            extra_width=extra_width,
            per_token=case.per_token,
            device=device,
        )
        if inputs.main_lengths is None:
            raise AssertionError("DSV4 exception case requires per-token lengths")
        _poison_inactive_topk_tails(
            inputs.main_index_scenarios,
            inputs.main_length_scenarios,
        )
        if case.has_extra:
            if (
                inputs.extra_cache is None
                or inputs.extra_indices is None
                or inputs.extra_index_scenarios is None
                or inputs.extra_lengths is None
                or inputs.extra_length_scenarios is None
            ):
                raise AssertionError("decode dual-cache inputs are incomplete")
            _poison_inactive_topk_tails(
                inputs.extra_index_scenarios,
                inputs.extra_length_scenarios,
            )
        expected = tuple(_reference(inputs, scenario)[0] for scenario in range(2))
        live_q = inputs.q
        kv_cache = inputs.main_cache
        live_indices = inputs.main_indices
        live_lengths = inputs.main_lengths
        index_scenarios = inputs.main_index_scenarios
        length_scenarios = inputs.main_length_scenarios
        q_scenarios = inputs.q_scenarios
        extra_cache = inputs.extra_cache
        extra_indices = inputs.extra_indices
        extra_lengths = inputs.extra_lengths
        extra_index_scenarios = inputs.extra_index_scenarios
        extra_length_scenarios = inputs.extra_length_scenarios
        sm_scale = _SM_SCALE

        def install(scenario: int) -> None:
            _install_scenario(inputs, scenario)

        immutable_tensors = {
            "kv_cache": kv_cache,
            "q_scenario_0": q_scenarios[0],
            "q_scenario_1": q_scenarios[1],
            "indices_scenario_0": index_scenarios[0],
            "indices_scenario_1": index_scenarios[1],
            "lengths_scenario_0": length_scenarios[0],
            "lengths_scenario_1": length_scenarios[1],
        }
        if extra_cache is not None:
            assert extra_index_scenarios is not None
            assert extra_length_scenarios is not None
            immutable_tensors.update(
                {
                    "extra_cache": extra_cache,
                    "extra_indices_scenario_0": extra_index_scenarios[0],
                    "extra_indices_scenario_1": extra_index_scenarios[1],
                    "extra_lengths_scenario_0": extra_length_scenarios[0],
                    "extra_lengths_scenario_1": extra_length_scenarios[1],
                }
            )
    else:
        if case.family != "glm" or case.has_extra:
            raise AssertionError(f"invalid decode case {case}")
        main_width = 128
        extra_width = 0
        inputs_glm = _make_glm_inputs(
            rows=rows,
            heads=heads,
            width=main_width,
            per_token=case.per_token,
            device=device,
        )
        if case.per_token:
            if inputs_glm.lengths is None:
                raise AssertionError("GLM per-token case is missing live lengths")
            _poison_inactive_topk_tails(
                inputs_glm.index_scenarios,
                inputs_glm.length_scenarios,
            )
        expected = tuple(
            _glm_reference(inputs_glm, scenario)[0] for scenario in range(2)
        )
        live_q = inputs_glm.q
        kv_cache = inputs_glm.launch_cache
        live_indices = inputs_glm.indices
        live_lengths = inputs_glm.lengths
        index_scenarios = inputs_glm.index_scenarios
        length_scenarios = inputs_glm.length_scenarios
        q_scenarios = inputs_glm.q_scenarios
        extra_cache = None
        extra_indices = None
        extra_lengths = None
        extra_index_scenarios = None
        extra_length_scenarios = None
        sm_scale = _GLM_SM_SCALE

        def install(scenario: int) -> None:
            _install_glm_scenario(inputs_glm, scenario)

        immutable_tensors = {
            "kv_cache": kv_cache,
            "packed_tokens": inputs_glm.packed_tokens,
            "q_scenario_0": q_scenarios[0],
            "q_scenario_1": q_scenarios[1],
            "indices_scenario_0": index_scenarios[0],
            "indices_scenario_1": index_scenarios[1],
            "lengths_scenario_0": length_scenarios[0],
            "lengths_scenario_1": length_scenarios[1],
        }

    if torch.allclose(expected[0], expected[1]):
        raise AssertionError("decode oracle scenarios are not distinct")
    workspace = _make_decode_workspace(
        # The exception manifests were compiled from a two-row fixed-capacity
        # workspace.  Keep that capacity (and therefore its split-axis stride)
        # for both dynamic row counts instead of silently replaying a different
        # compile-spec layout.
        rows=2,
        heads=heads,
        width=main_width + extra_width,
        family=case.family,
        device=device,
    )
    output = torch.empty(
        (rows, heads, _GLM_V_DIM),
        dtype=torch.bfloat16,
        device=device,
    )
    stable_tensors = {
        "q": live_q,
        "kv_cache": kv_cache,
        "indices": live_indices,
        "output": output,
        "workspace": workspace.shared_scratch,
    }
    if live_lengths is not None:
        stable_tensors["lengths"] = live_lengths
    if extra_cache is not None:
        assert extra_indices is not None
        assert extra_lengths is not None
        stable_tensors.update(
            {
                "extra_cache": extra_cache,
                "extra_indices": extra_indices,
                "extra_lengths": extra_lengths,
            }
        )
    stable_pointers = {
        name: tensor.data_ptr() for name, tensor in stable_tensors.items()
    }
    immutable_before = {
        name: _tensor_sha256(tensor) for name, tensor in immutable_tensors.items()
    }

    def launch(label: str) -> torch.Tensor:
        select_arm(label)
        return mla_kernel.run_unified_decode(
            q_all=live_q,
            swa_k_cache=kv_cache,
            swa_indices=live_indices,
            swa_topk_lengths=live_lengths,
            workspace=workspace,
            sm_scale=sm_scale,
            swa_page_size=_PAGE_SIZE,
            indexed_k_cache=extra_cache,
            indexed_indices=extra_indices,
            indexed_topk_lengths=extra_lengths,
            indexed_page_size=_PAGE_SIZE if extra_cache is not None else None,
            forced_num_splits=1,
            out=output,
        )

    stream = torch.cuda.Stream(device=device)
    graphs: dict[str, torch.cuda.CUDAGraph] = {}
    topologies: dict[str, dict[str, object]] = {}
    dispatch_before = {label: dict(dispatch_counts[label]) for label in labels}
    with torch.cuda.stream(stream):
        install(0)
    stream.synchronize()
    for label in labels:
        with torch.cuda.stream(stream):
            warm_output = launch(label)
        stream.synchronize()
        if warm_output.data_ptr() != output.data_ptr():
            raise AssertionError(f"{label}: launcher replaced caller-owned output")
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph, stream=stream):
            launch(label)
        stream.synchronize()
        graphs[label] = graph
        topologies[label] = _graph_topology(graph)

    if topologies[labels[0]] != topologies[labels[1]]:
        raise AssertionError("decode arm graph topologies differ")
    topology = topologies[labels[0]]
    if topology["node_count"] != 2 or topology["kernel_node_count"] != 2:
        raise AssertionError(f"unexpected decode graph topology: {topology}")

    correctness: dict[str, dict[str, object]] = {}

    def validate(stage: str, scenario: int) -> None:
        with torch.cuda.stream(stream):
            install(scenario)
        stream.synchronize()
        if not torch.equal(live_indices, index_scenarios[scenario]):
            raise AssertionError("live decode indices do not match installed scenario")
        if live_lengths is not None:
            if not torch.equal(live_lengths, length_scenarios[scenario]):
                raise AssertionError(
                    "live decode lengths do not match installed scenario"
                )
            for row in range(rows):
                length = int(live_lengths[row].item())
                if not 0 < length <= main_width:
                    raise AssertionError(f"row {row}: invalid main length {length}")
                if length < main_width and int(live_indices[row, length].item()) != -1:
                    raise AssertionError(
                        f"row {row}: inactive main tail is not poisoned"
                    )
        if extra_indices is not None:
            assert extra_lengths is not None
            assert extra_index_scenarios is not None
            assert extra_length_scenarios is not None
            if not torch.equal(extra_indices, extra_index_scenarios[scenario]):
                raise AssertionError(
                    "live extra indices do not match installed scenario"
                )
            if not torch.equal(extra_lengths, extra_length_scenarios[scenario]):
                raise AssertionError(
                    "live extra lengths do not match installed scenario"
                )
            for row in range(rows):
                length = int(extra_lengths[row].item())
                if not 0 < length <= extra_width:
                    raise AssertionError(f"row {row}: invalid extra length {length}")
                if (
                    length < extra_width
                    and int(extra_indices[row, length].item()) != -1
                ):
                    raise AssertionError(
                        f"row {row}: inactive extra tail is not poisoned"
                    )
        _stable_pointer_check(stable_tensors, stable_pointers)
        arm_outputs: dict[str, torch.Tensor] = {}
        stage_result: dict[str, object] = {"scenario": scenario, "arms": {}}
        for label in labels:
            with torch.cuda.stream(stream):
                output.fill_(float("nan"))
            stream.synchronize()
            allocator_before = _allocator_counters(device)
            with torch.cuda.stream(stream):
                graphs[label].replay()
            stream.synchronize()
            allocator_after = _allocator_counters(device)
            if allocator_after != allocator_before:
                raise AssertionError(
                    f"{label}: replay allocated: {allocator_before} -> {allocator_after}"
                )
            _stable_pointer_check(stable_tensors, stable_pointers)
            _assert_output(
                output,
                expected[scenario],
                label=f"{label} {stage} {case.name} rows={rows} scenario={scenario}",
            )
            arm_outputs[label] = output.clone()
            stage_result["arms"][label] = {
                "output": _correctness_metrics(output, expected[scenario]),
                "allocator_before": allocator_before,
                "allocator_after": allocator_after,
                "passed": True,
            }
        if not torch.equal(arm_outputs[labels[0]], arm_outputs[labels[1]]):
            raise AssertionError(f"{stage}: decode arms are not bit exact")
        stage_result.update({"output_arms_bit_exact": True, "passed": True})
        correctness[stage] = stage_result

    validate("pre", 0)
    timed_live_tensors = {
        "q": live_q,
        "indices": live_indices,
    }
    if live_lengths is not None:
        timed_live_tensors["lengths"] = live_lengths
    if extra_indices is not None:
        assert extra_lengths is not None
        timed_live_tensors.update(
            {"extra_indices": extra_indices, "extra_lengths": extra_lengths}
        )
    timed_live_before = {
        name: _tensor_sha256(tensor) for name, tensor in timed_live_tensors.items()
    }
    timing = _run_timing(
        graphs=graphs,
        labels=labels,
        all_graph_spec_hashes=(case.decode_spec_hash, case.merge_spec_hash),
        precondition_replays=precondition_replays,
        precondition_seconds=precondition_seconds,
        maximum_precondition_seconds=maximum_precondition_seconds,
        warmup_cycles=warmup_cycles,
        cycles=cycles,
        event_batch_cycles=event_batch_cycles,
        replays_per_reported_sample=replays_per_reported_sample,
        l2_flush_bytes=l2_flush_bytes,
        expected_physical_gpu=expected_physical_gpu,
        max_sm_clock_delta_mhz=max_sm_clock_delta_mhz,
        stream=stream,
        device=device,
    )
    timed_live_after = {
        name: _tensor_sha256(tensor) for name, tensor in timed_live_tensors.items()
    }
    if timed_live_after != timed_live_before:
        raise AssertionError("live decode timing input changed during replay")
    validate("post", 1)
    _stable_pointer_check(stable_tensors, stable_pointers)
    immutable_after = {
        name: _tensor_sha256(tensor) for name, tensor in immutable_tensors.items()
    }
    if immutable_after != immutable_before:
        raise AssertionError("read-only decode input changed")
    dispatch_after = {label: dict(dispatch_counts[label]) for label in labels}
    expected_specs = (case.decode_spec_hash, case.merge_spec_hash)
    return {
        "rows": rows,
        "shape": {
            "family": case.family,
            "heads": heads,
            "main_topk": main_width,
            "extra_topk": extra_width,
            "per_token_lengths": case.per_token,
            "forced_num_splits": 1,
        },
        "workspace": {
            "caller_owned_bytes": workspace.shared_scratch.numel(),
            "max_chunks_per_row": workspace.max_chunks_per_row,
            "fixed": True,
            "preplanned": True,
        },
        "graph": {
            "cuda_graph_replay": True,
            "same_addresses_across_arms": True,
            "stable_pointers": stable_pointers,
            "topologies": topologies,
            "topologies_equal": True,
            "allocator_counters": list(_ALLOCATOR_COUNTERS),
            "dispatch_count_delta": {
                label: {
                    spec_hash: (
                        dispatch_after[label][spec_hash]
                        - dispatch_before[label][spec_hash]
                    )
                    for spec_hash in expected_specs
                }
                for label in labels
            },
        },
        "correctness": correctness,
        "poisoned_outputs_overwritten": True,
        "live_input_mutation_changed_output": True,
        "read_only_inputs": {
            "sha256_before": immutable_before,
            "sha256_after": immutable_after,
            "timed_live_scenario_0": {
                "sha256_before": timed_live_before,
                "sha256_after": timed_live_after,
                "unchanged": True,
            },
            "unchanged": True,
        },
        "timing": timing,
    }


def _run_merge_rows(
    *,
    rows: int,
    case: MergeCase,
    labels: tuple[str, str],
    select_arm: Callable[[str], None],
    dispatch_counts: dict[str, dict[str, int]],
    precondition_replays: int,
    precondition_seconds: float,
    maximum_precondition_seconds: float,
    warmup_cycles: int,
    cycles: int,
    event_batch_cycles: int,
    replays_per_reported_sample: int,
    l2_flush_bytes: int,
    expected_physical_gpu: int,
    max_sm_clock_delta_mhz: float,
    device: torch.device,
) -> dict[str, object]:
    base_problem = _make_fixed_merge_problem(
        rows=rows,
        heads=case.heads,
        chunks=case.chunks,
        device=device,
    )
    # Keep the production shapes/strides while separating output from chunk zero.
    # This makes every replay consume immutable partials instead of feeding the
    # previous merged output back into the next launch.
    problem = replace(base_problem, output=torch.empty_like(base_problem.output))
    scenarios = _make_merge_scenarios(
        rows=rows,
        heads=case.heads,
        chunks=case.chunks,
        device=device,
    )
    sink_scenarios: tuple[torch.Tensor, torch.Tensor] | None = None
    live_sink: torch.Tensor | None = None
    if case.with_sink:
        sink_scenarios = (
            torch.linspace(-1.25, 0.75, case.heads, dtype=torch.float32, device=device),
            torch.linspace(0.9, -0.6, case.heads, dtype=torch.float32, device=device),
        )
        live_sink = torch.empty_like(sink_scenarios[0])
    expected = tuple(
        _split_merge_fp32_oracle(
            partials,
            lse,
            chunks=case.chunks,
            attn_sink=(None if sink_scenarios is None else sink_scenarios[index]),
        )
        for index, (partials, lse) in enumerate(scenarios)
    )
    if torch.allclose(expected[0], expected[1]):
        raise AssertionError("merge oracle scenarios are not distinct")
    binding = mla_merge.build_sparse_mla_split_decode_merge_binding(
        tmp_output=problem.tmp_output,
        tmp_lse=problem.tmp_lse,
        num_chunks_ptr=problem.num_chunks_ptr,
        output=problem.output,
        num_chunks=case.chunks,
        attn_sink=live_sink,
    )
    stable_tensors = {
        "tmp_output": problem.tmp_output,
        "tmp_lse": problem.tmp_lse,
        "num_chunks": problem.num_chunks_ptr,
        "output": problem.output,
    }
    if live_sink is not None:
        stable_tensors["attn_sink"] = live_sink
    stable_pointers = {
        name: tensor.data_ptr() for name, tensor in stable_tensors.items()
    }
    immutable_tensors = {
        f"partials_scenario_{index}": scenario[0]
        for index, scenario in enumerate(scenarios)
    }
    immutable_tensors.update(
        {
            f"lse_scenario_{index}": scenario[1]
            for index, scenario in enumerate(scenarios)
        }
    )
    if sink_scenarios is not None:
        immutable_tensors.update(
            {
                f"sink_scenario_{index}": sink
                for index, sink in enumerate(sink_scenarios)
            }
        )
    immutable_before = {
        name: _tensor_sha256(tensor) for name, tensor in immutable_tensors.items()
    }

    def install(scenario: int) -> None:
        _install_merge_scenario(
            problem,
            partials=scenarios[scenario][0],
            lse=scenarios[scenario][1],
            live_sink=live_sink,
            source_sink=(None if sink_scenarios is None else sink_scenarios[scenario]),
        )

    def launch(label: str) -> None:
        select_arm(label)
        binding.run()

    stream = torch.cuda.Stream(device=device)
    graphs: dict[str, torch.cuda.CUDAGraph] = {}
    topologies: dict[str, dict[str, object]] = {}
    dispatch_before = {label: dict(dispatch_counts[label]) for label in labels}
    with torch.cuda.stream(stream):
        install(0)
    stream.synchronize()
    for label in labels:
        with torch.cuda.stream(stream):
            launch(label)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph, stream=stream):
            launch(label)
        stream.synchronize()
        graphs[label] = graph
        topologies[label] = _graph_topology(graph)

    if topologies[labels[0]] != topologies[labels[1]]:
        raise AssertionError("merge arm graph topologies differ")
    topology = topologies[labels[0]]
    if topology["node_count"] != 1 or topology["kernel_node_count"] != 1:
        raise AssertionError(f"unexpected merge graph topology: {topology}")

    correctness: dict[str, dict[str, object]] = {}

    def validate(stage: str, scenario: int) -> None:
        with torch.cuda.stream(stream):
            install(scenario)
        stream.synchronize()
        _stable_pointer_check(stable_tensors, stable_pointers)
        arm_outputs: dict[str, torch.Tensor] = {}
        stage_result: dict[str, object] = {"scenario": scenario, "arms": {}}
        for label in labels:
            with torch.cuda.stream(stream):
                problem.output.fill_(float("nan"))
            stream.synchronize()
            allocator_before = _allocator_counters(device)
            with torch.cuda.stream(stream):
                graphs[label].replay()
            stream.synchronize()
            allocator_after = _allocator_counters(device)
            if allocator_after != allocator_before:
                raise AssertionError(
                    f"{label}: replay allocated: {allocator_before} -> {allocator_after}"
                )
            _stable_pointer_check(stable_tensors, stable_pointers)
            if not bool(torch.isfinite(problem.output).all().item()):
                raise AssertionError(f"{label}: non-finite merge output")
            if int(torch.count_nonzero(problem.output).item()) == 0:
                raise AssertionError(f"{label}: zero merge output")
            torch.testing.assert_close(
                problem.output.float(),
                expected[scenario],
                atol=1.5e-2,
                rtol=1.5e-2,
            )
            arm_outputs[label] = problem.output.clone()
            stage_result["arms"][label] = {
                "output": _correctness_metrics(problem.output, expected[scenario]),
                "allocator_before": allocator_before,
                "allocator_after": allocator_after,
                "passed": True,
            }
        if not torch.equal(arm_outputs[labels[0]], arm_outputs[labels[1]]):
            raise AssertionError(f"{stage}: merge arms are not bit exact")
        stage_result.update({"output_arms_bit_exact": True, "passed": True})
        correctness[stage] = stage_result

    validate("pre", 0)
    timed_live_tensors = {
        "tmp_output": problem.tmp_output,
        "tmp_lse": problem.tmp_lse,
        "num_chunks": problem.num_chunks_ptr,
    }
    if live_sink is not None:
        timed_live_tensors["attn_sink"] = live_sink
    timed_live_before = {
        name: _tensor_sha256(tensor) for name, tensor in timed_live_tensors.items()
    }
    timing = _run_timing(
        graphs=graphs,
        labels=labels,
        all_graph_spec_hashes=(case.spec_hash,),
        precondition_replays=precondition_replays,
        precondition_seconds=precondition_seconds,
        maximum_precondition_seconds=maximum_precondition_seconds,
        warmup_cycles=warmup_cycles,
        cycles=cycles,
        event_batch_cycles=event_batch_cycles,
        replays_per_reported_sample=replays_per_reported_sample,
        l2_flush_bytes=l2_flush_bytes,
        expected_physical_gpu=expected_physical_gpu,
        max_sm_clock_delta_mhz=max_sm_clock_delta_mhz,
        stream=stream,
        device=device,
    )
    timed_live_after = {
        name: _tensor_sha256(tensor) for name, tensor in timed_live_tensors.items()
    }
    if timed_live_after != timed_live_before:
        raise AssertionError("live merge timing input changed during replay")
    validate("post", 1)
    _stable_pointer_check(stable_tensors, stable_pointers)
    immutable_after = {
        name: _tensor_sha256(tensor) for name, tensor in immutable_tensors.items()
    }
    if immutable_after != immutable_before:
        raise AssertionError("read-only merge input changed")
    dispatch_after = {label: dict(dispatch_counts[label]) for label in labels}
    workspace_bytes = sum(
        tensor.untyped_storage().nbytes()
        for tensor in (
            problem.tmp_output,
            problem.tmp_lse,
            problem.num_chunks_ptr,
            problem.output,
        )
    )
    return {
        "rows": rows,
        "shape": {
            "heads": case.heads,
            "chunks": case.chunks,
            "head_dim": int(problem.output.shape[-1]),
            "with_sink": case.with_sink,
            "output_aliases_chunk_zero": False,
        },
        "workspace": {
            "caller_owned_bytes_upper_bound": workspace_bytes,
            "fixed": True,
            "preplanned": True,
        },
        "graph": {
            "cuda_graph_replay": True,
            "same_addresses_across_arms": True,
            "stable_pointers": stable_pointers,
            "topologies": topologies,
            "topologies_equal": True,
            "allocator_counters": list(_ALLOCATOR_COUNTERS),
            "dispatch_count_delta": {
                label: {
                    case.spec_hash: (
                        dispatch_after[label][case.spec_hash]
                        - dispatch_before[label][case.spec_hash]
                    )
                }
                for label in labels
            },
        },
        "correctness": correctness,
        "poisoned_outputs_overwritten": True,
        "live_input_mutation_changed_output": True,
        "read_only_inputs": {
            "sha256_before": immutable_before,
            "sha256_after": immutable_after,
            "timed_live_scenario_0": {
                "sha256_before": timed_live_before,
                "sha256_after": timed_live_after,
                "unchanged": True,
            },
            "unchanged": True,
        },
        "timing": timing,
    }


def main() -> None:
    args = _args()
    case = _CASES[args.case]
    rows = _row_sweep(args.rows, case)
    if args.a_label == args.b_label:
        raise ValueError("A/B labels must differ")
    if args.cycles < 500 or args.cycles % 2:
        raise ValueError("--cycles must be an even integer of at least 500")
    if args.precondition_replays <= 0:
        raise ValueError("--precondition-replays must be positive")
    if args.warmup_cycles <= 0 or args.event_batch_cycles <= 0:
        raise ValueError("warmup and event-batch cycles must be positive")
    if args.replays_per_reported_sample <= 0:
        raise ValueError("--replays-per-reported-sample must be positive")
    if not math.isfinite(args.precondition_seconds) or args.precondition_seconds < 5.0:
        raise ValueError("--precondition-seconds must be finite and at least 5")
    if (
        not math.isfinite(args.maximum_precondition_seconds)
        or args.maximum_precondition_seconds < args.precondition_seconds
        or args.maximum_precondition_seconds > 60.0
    ):
        raise ValueError(
            "--maximum-precondition-seconds must cover the minimum and be at most 60"
        )
    if (
        not math.isfinite(args.max_sm_clock_delta_mhz)
        or args.max_sm_clock_delta_mhz <= 0.0
        or args.max_sm_clock_delta_mhz > 60.0
    ):
        raise ValueError("--max-sm-clock-delta-mhz must be in (0, 60]")
    if args.l2_flush_bytes < 0:
        raise ValueError("--l2-flush-bytes must be non-negative")

    gpu_scope = require_target_gpu(args.expected_physical_gpu)
    device = torch.device("cuda", 0)
    torch.empty((1,), dtype=torch.uint8, device=device)
    torch.cuda.synchronize(device)

    labels = (args.a_label, args.b_label)
    a_keys = _key_overrides(args.a_key)
    b_keys = _key_overrides(args.b_key)
    if isinstance(case, DecodeCase):
        spec_hashes = (case.decode_spec_hash, case.merge_spec_hash)

        def validator(spec_hash: str) -> Callable[[dict[str, Any]], None]:
            if spec_hash == case.decode_spec_hash:
                return lambda manifest: _validate_decode_manifest(manifest, case)
            return lambda manifest: _validate_merge_manifest(
                manifest,
                spec_hash=spec_hash,
                static_num_chunks=1,
                with_sink=False,
            )

    else:
        spec_hashes = (case.spec_hash,)

        def validator(spec_hash: str) -> Callable[[dict[str, Any]], None]:
            return lambda manifest: _validate_merge_manifest(
                manifest,
                spec_hash=spec_hash,
                static_num_chunks=case.chunks,
                with_sink=case.with_sink,
            )

    unexpected_a_keys = sorted(set(a_keys) - set(spec_hashes))
    unexpected_b_keys = sorted(set(b_keys) - set(spec_hashes))
    if unexpected_a_keys or unexpected_b_keys:
        raise ValueError(
            f"cache-key overrides do not belong to {case.name}: "
            f"a={unexpected_a_keys}, b={unexpected_b_keys}"
        )

    compiled: dict[str, dict[str, Any]] = {label: {} for label in labels}
    provenance: dict[str, dict[str, dict[str, Any]]] = {label: {} for label in labels}
    for label, cache, keys in (
        (labels[0], args.a_cache, a_keys),
        (labels[1], args.b_cache, b_keys),
    ):
        for spec_hash in spec_hashes:
            loaded, record = _load_exact(
                cache,
                spec_hash,
                keys.get(spec_hash),
                validator(spec_hash),
            )
            compiled[label][spec_hash] = loaded
            provenance[label][spec_hash] = record

    for spec_hash in spec_hashes:
        a_record = provenance[labels[0]][spec_hash]
        b_record = provenance[labels[1]][spec_hash]
        if a_record["compile_spec_json"] != b_record["compile_spec_json"]:
            raise RuntimeError(f"A/B compile specs differ for {spec_hash}")
        if a_record["kernel_id"] != b_record["kernel_id"]:
            raise RuntimeError(f"A/B kernel IDs differ for {spec_hash}")

    integrity_initial = {
        label: {
            spec_hash: _verify_artifact(provenance[label][spec_hash])
            for spec_hash in spec_hashes
        }
        for label in labels
    }
    l2_flush_bytes = resolve_l2_flush_bytes(args.l2_flush_bytes)
    if make_l2_flush_fn(True, l2_flush_bytes) is None:
        raise AssertionError("cold-L2 flush function was not constructed")
    torch.cuda.synchronize(device)
    gpu_mode_initial = gpu_mode_snapshot(args.expected_physical_gpu)

    active_label: list[str | None] = [None]
    dispatch_counts = {
        label: {spec_hash: 0 for spec_hash in spec_hashes} for label in labels
    }
    observed_specs: set[str] = set()
    original_decode_launch = mla_kernel.b12x_launch
    original_merge_launch = mla_merge.b12x_launch
    force_glm_h8_native = isinstance(case, DecodeCase) and case.family == "glm"
    previous_glm_h8_native = os.environ.get(_GLM_H8_NATIVE_ENV)

    def exact_dispatch(
        _func,
        *,
        compile_spec,
        compile_args,
        runtime_args,
        compile_kwargs=None,
    ):
        del _func, compile_args, compile_kwargs
        label = active_label[0]
        if label not in compiled:
            raise RuntimeError(f"no exact-object arm selected: {label!r}")
        spec_hash = compile_spec.hash_key
        if spec_hash not in compiled[label]:
            raise RuntimeError(f"launcher produced unexpected spec {spec_hash}")
        observed_specs.add(spec_hash)
        dispatch_counts[label][spec_hash] += 1
        return cute_compiler.run_compiled(compiled[label][spec_hash], runtime_args)

    if force_glm_h8_native:
        os.environ[_GLM_H8_NATIVE_ENV] = "1"
    mla_kernel.b12x_launch = exact_dispatch
    mla_merge.b12x_launch = exact_dispatch
    row_results: list[dict[str, object]] = []
    integrity_by_rows: dict[str, dict[str, object]] = {}
    try:
        for row_count in rows:
            integrity_before = {
                label: {
                    spec_hash: _verify_artifact(provenance[label][spec_hash])
                    for spec_hash in spec_hashes
                }
                for label in labels
            }

            def select_arm(label: str) -> None:
                active_label[0] = label

            common = {
                "rows": row_count,
                "case": case,
                "labels": labels,
                "select_arm": select_arm,
                "dispatch_counts": dispatch_counts,
                "precondition_replays": args.precondition_replays,
                "precondition_seconds": args.precondition_seconds,
                "maximum_precondition_seconds": args.maximum_precondition_seconds,
                "warmup_cycles": args.warmup_cycles,
                "cycles": args.cycles,
                "event_batch_cycles": args.event_batch_cycles,
                "replays_per_reported_sample": args.replays_per_reported_sample,
                "l2_flush_bytes": l2_flush_bytes,
                "expected_physical_gpu": args.expected_physical_gpu,
                "max_sm_clock_delta_mhz": args.max_sm_clock_delta_mhz,
                "device": device,
            }
            if isinstance(case, DecodeCase):
                row_result = _run_decode_rows(**common)
            else:
                row_result = _run_merge_rows(**common)
            integrity_after = {
                label: {
                    spec_hash: _verify_artifact(provenance[label][spec_hash])
                    for spec_hash in spec_hashes
                }
                for label in labels
            }
            if integrity_after != integrity_before:
                raise RuntimeError(
                    f"artifact integrity changed while benchmarking rows={row_count}"
                )
            integrity_by_rows[str(row_count)] = {
                "before": integrity_before,
                "after": integrity_after,
            }
            row_results.append(row_result)
            active_label[0] = None
    finally:
        mla_kernel.b12x_launch = original_decode_launch
        mla_merge.b12x_launch = original_merge_launch
        if force_glm_h8_native:
            if previous_glm_h8_native is None:
                os.environ.pop(_GLM_H8_NATIVE_ENV, None)
            else:
                os.environ[_GLM_H8_NATIVE_ENV] = previous_glm_h8_native
        active_label[0] = None

    if observed_specs != set(spec_hashes):
        raise AssertionError(f"unexpected observed specs: {observed_specs}")
    integrity_final = {
        label: {
            spec_hash: _verify_artifact(provenance[label][spec_hash])
            for spec_hash in spec_hashes
        }
        for label in labels
    }
    if integrity_final != integrity_initial:
        raise RuntimeError("artifact integrity changed during benchmark")
    gpu_mode_final = gpu_mode_snapshot(args.expected_physical_gpu)
    case_record: dict[str, object] = {
        "name": case.name,
        "kind": "decode" if isinstance(case, DecodeCase) else "merge",
        "compile_spec_hashes": list(spec_hashes),
        "rows": list(rows),
    }
    if isinstance(case, DecodeCase):
        case_record.update(
            {
                "family": case.family,
                "per_token_lengths": case.per_token,
                "has_extra": case.has_extra,
            }
        )
    else:
        case_record.update(
            {
                "with_sink": case.with_sink,
                "heads": case.heads,
                "chunks": case.chunks,
            }
        )

    result: dict[str, object] = {
        "schema": "b12x.attention.mla.decode_merge.exact_cache_abba.v2",
        "evidence_status": args.evidence_status,
        "command": [sys.executable, *sys.argv],
        "case": case_record,
        "labels": {"a": labels[0], "b": labels[1]},
        "artifacts": provenance,
        "artifact_integrity": {
            "initial": integrity_initial,
            "by_rows": integrity_by_rows,
            "final": integrity_final,
            "immutable": True,
        },
        "gpu": {
            "cuda_visible_devices": gpu_scope["visible_devices"],
            "physical_index": gpu_scope["physical_index"],
            "visible_ordinal": gpu_scope["logical_device"],
            "uuid": gpu_scope["uuid"],
            "name": gpu_scope["name"],
            "capability": gpu_scope["capability"],
            "mode_initial": gpu_mode_initial,
            "mode_final": gpu_mode_final,
        },
        "runtime_contract": {
            "cuda_graph_replay": True,
            "same_addresses_across_arms": True,
            "fixed_preplanned_workspace": True,
            "warm_and_cold_l2": True,
            "l2_flush_bytes": l2_flush_bytes,
            "minimum_cycles_enforced": 500,
            "even_cycles_required": True,
            "minimum_balanced_precondition_seconds": args.precondition_seconds,
            "maximum_balanced_precondition_seconds": (
                args.maximum_precondition_seconds
            ),
            "replays_per_reported_sample": args.replays_per_reported_sample,
            "required_pstate": "P1",
            "max_sm_clock_delta_mhz": args.max_sm_clock_delta_mhz,
            "active_clock_throttle_reasons_required": 0,
            "dispatch_env_override": (
                {_GLM_H8_NATIVE_ENV: "1"} if force_glm_h8_native else {}
            ),
        },
        "rows": row_results,
        "observed_compile_specs": sorted(observed_specs),
        "dispatch_counts_during_setup_only": dispatch_counts,
        "provenance": {
            "git_commit": _git_output("rev-parse", "HEAD"),
            "git_worktree": _git_output("rev-parse", "--show-toplevel"),
            "git_status_short": _git_output("status", "--short").splitlines(),
            "source_sha256": {
                "benchmark": _sha256_file(Path(__file__).resolve()),
                "shared_abba": _sha256_file(
                    PACKAGE_ROOT / "diagnostics" / "paired" / "mla_prefill_mg.py"
                ),
                "decode": _sha256_file(REPO_ROOT / "b12x/attention/mla/kernel.py"),
                "merge": _sha256_file(REPO_ROOT / "b12x/attention/mla/merge.py"),
                "compiler": _sha256_file(REPO_ROOT / "b12x/cute/compiler.py"),
                "corpus_helpers": _sha256_file(
                    REPO_ROOT / "tests/test_attention_mla_unified_corpus.py"
                ),
                "merge_oracle": _sha256_file(
                    REPO_ROOT / "tests/test_attention_mla_merge.py"
                ),
                "corpus_matrix": _sha256_file(
                    DATA_ROOT / "cute_migration_corpus_matrix.json"
                ),
            },
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "python_executable": sys.executable,
            "python_prefix": sys.prefix,
        },
    }
    result["result_sha256"] = _json_sha256(result)
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
                "gpu": result["gpu"],
                "ratios_b_over_a": {
                    str(row["rows"]): {
                        condition: data["timings"]["ratios_b_over_a"]
                        for condition, data in row["timing"]["conditions"].items()
                    }
                    for row in row_results
                },
                "result_sha256": result["result_sha256"],
            },
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
