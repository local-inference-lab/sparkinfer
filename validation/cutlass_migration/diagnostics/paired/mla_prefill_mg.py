#!/usr/bin/env python3
"""Exact-object, same-address CUDA-graph ABBA for SM120 MLA MG prefill.

This benchmark compares frozen CUTLASS-DSL cache objects without recompiling
them.  Its case table covers every MG-prefill specialization in the CUTLASS 4.6
register/resource exception set: both DSV4 compute modes and head groups, the
dual-cache arm, both GLM BF16-cache head groups, both GLM NVFP4 head groups,
and the sink/no-sink DSV4 BF16 variants.

Each row count receives one fixed set of inputs, outputs, and caller-owned
workspace.  Both objects are captured against those same addresses.  Oracle
and exact arm-equality checks run before and after balanced warm/cold-L2 ABBA
timing, and the cached object digests are checked before and after every row.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Callable

import torch
from cuda.bindings import driver as cuda_driver

from benchmarks.common import make_l2_flush_fn, resolve_l2_flush_bytes
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
from validation.cutlass_migration.paths import DATA_ROOT, REPO_ROOT
import b12x.attention.mla.prefill_mg as prefill_mg
import b12x.cute.compiler as cute_compiler
from b12x.attention.mla.kernel import run_unified_prefill
from b12x.attention.mla.traits import ScaleFormat
from tests.test_attention_mla_unified_corpus import (
    _ALLOCATOR_COUNTERS,
    _GLM_V_DIM,
    _GLM_SM_SCALE,
    _PAGE_SIZE,
    _SM_SCALE,
    _allocator_counters,
    _assert_output,
    _assert_prefill_boundary_heads,
    _install_scenario,
    _install_glm_scenario,
    _install_nvfp4_glm_scenario,
    _make_inputs,
    _make_glm_inputs,
    _make_nvfp4_glm_inputs,
    _glm_reference,
    _nvfp4_glm_reference,
    _poison_inactive_topk_tails,
    _reference,
)


_MG_GATE_ENV = "B12X_MLA_SM120_PREFILL_MG"
_DEFAULT_ROWS = (1, 2, 8, 32, 128, 512, 2048)


@dataclass(frozen=True)
class PrefillSpec:
    name: str
    spec_hash: str
    family: str
    heads: int
    topk: int
    n_hg: int
    compute_mode: int
    has_sink: bool
    extra_topk: int = 0
    heads_per_cta: int | None = None
    valid_hpb: int = 16
    pack_hilo_rows: int = 0

    @property
    def model_type(self) -> int:
        return 0 if self.family == "dsv4" else 1

    @property
    def scale_format(self) -> int:
        if self.family == "dsv4":
            return 0
        if self.family == "glm":
            return 1
        if self.family == "glm-nvfp4":
            return int(ScaleFormat.NVFP4_E4M3)
        raise AssertionError(f"unknown MLA family {self.family!r}")


_SPECS = {
    spec.spec_hash: spec
    for spec in (
        PrefillSpec(
            name="dsv4-fp8-hg2-h32-topk512",
            spec_hash=(
                "08e456f5a5daca77e6036fb8613c733d2e929c78b12ecbf8260faa07bd10c334"
            ),
            family="dsv4",
            heads=32,
            topk=512,
            n_hg=2,
            compute_mode=0,
            has_sink=False,
        ),
        PrefillSpec(
            name="dsv4-bf16-hg1-h16-topk128-sink",
            spec_hash=(
                "39b912c432760c48e04b7f3fda79b8810b316732a5e248c8a34d975a38dc8653"
            ),
            family="dsv4",
            heads=16,
            topk=128,
            n_hg=1,
            compute_mode=1,
            has_sink=True,
        ),
        PrefillSpec(
            name="dsv4-bf16-hg1-h16-topk128-no-sink",
            spec_hash=(
                "83283d9b086d44c3d228643246f7041fb30acba6080381873bd001d19a08cf90"
            ),
            family="dsv4",
            heads=16,
            topk=128,
            n_hg=1,
            compute_mode=1,
            has_sink=False,
        ),
        PrefillSpec(
            name="dsv4-fp8-hg1-h16-topk512",
            spec_hash=(
                "57d0b16e1c98aadb68c48b8913aa9486aeeee88f8a6ec3118061aa6cf9cb56fa"
            ),
            family="dsv4",
            heads=16,
            topk=512,
            n_hg=1,
            compute_mode=0,
            has_sink=False,
        ),
        PrefillSpec(
            name="dsv4-bf16-hg2-h32-topk128",
            spec_hash=(
                "ea673f66ef9d6ac8bc145270f4c191851e31cea39ce4dbd8060fc5df0f3f5ce5"
            ),
            family="dsv4",
            heads=32,
            topk=128,
            n_hg=2,
            compute_mode=1,
            has_sink=False,
        ),
        PrefillSpec(
            name="dsv4-bf16-hg1-h16-topk128-extra64-sink",
            spec_hash=(
                "f0733c7befb15907efc9254bc5dbe57d481314133add47ed771cf20bba64c4c1"
            ),
            family="dsv4",
            heads=16,
            topk=128,
            n_hg=1,
            compute_mode=1,
            has_sink=True,
            extra_topk=64,
        ),
        PrefillSpec(
            name="glm-fp8-hg1-h16-topk512",
            spec_hash=(
                "5dbb403072ad198b4d55c430c4cde20e1d9b1a61ad6aebe0b041b0aaf8e6e61b"
            ),
            family="glm",
            heads=16,
            topk=512,
            n_hg=1,
            compute_mode=0,
            has_sink=False,
        ),
        PrefillSpec(
            name="glm-fp8-hg2-h32-topk512",
            spec_hash=(
                "cbec33749ceac101c43b1b3e6648f900363fec59504666102ec4bc8643ba4166"
            ),
            family="glm",
            heads=32,
            topk=512,
            n_hg=2,
            compute_mode=0,
            has_sink=False,
        ),
        PrefillSpec(
            name="glm-nvfp4-bf16-hg1-h16-topk512",
            spec_hash=(
                "8eea86da9a3bc76c2cc2fdba7fbc4ed12895d146a6e2930cbdee3e45ed30f18e"
            ),
            family="glm-nvfp4",
            heads=16,
            topk=512,
            n_hg=1,
            compute_mode=1,
            has_sink=False,
        ),
        PrefillSpec(
            name="glm-nvfp4-bf16-hg2-h32-topk512",
            spec_hash=(
                "4758dee1c901414968459568b0a8db2c1217a11bebee69a0badba07c9311171e"
            ),
            family="glm-nvfp4",
            heads=32,
            topk=512,
            n_hg=2,
            compute_mode=1,
            has_sink=False,
        ),
        PrefillSpec(
            name="glm-fp8-hg1-h8-topk512-packed-hilo",
            spec_hash=(
                "387817cbff4742f5730d25b0d1687160e91967b0d3b950a4aaa3af761bed038b"
            ),
            family="glm",
            heads=8,
            topk=512,
            n_hg=1,
            compute_mode=0,
            has_sink=False,
            heads_per_cta=16,
            valid_hpb=8,
            pack_hilo_rows=1,
        ),
    )
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_evidence_status_argument(parser)
    add_target_gpu_argument(parser)
    parser.add_argument("--a-cache", type=Path, required=True)
    parser.add_argument("--a-key", help="optional cache-key disambiguator")
    parser.add_argument("--a-label", default="cutlass-4.5.2")
    parser.add_argument("--b-cache", type=Path, required=True)
    parser.add_argument("--b-key", help="optional cache-key disambiguator")
    parser.add_argument("--b-label", default="cutlass-4.6.0")
    parser.add_argument(
        "--spec-hash",
        required=True,
        choices=tuple(_SPECS),
        help="frozen compile-spec hash to replay",
    )
    parser.add_argument(
        "--rows",
        action="append",
        default=[],
        metavar="N[,N...]",
        help=(
            "row counts to benchmark; repeat or use comma-separated values "
            f"(default: {','.join(map(str, _DEFAULT_ROWS))})"
        ),
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


def _row_sweep(raw_values: list[str]) -> tuple[int, ...]:
    if not raw_values:
        return _DEFAULT_ROWS
    rows: list[int] = []
    for raw in raw_values:
        for value in raw.split(","):
            try:
                row = int(value.strip())
            except ValueError as error:
                raise ValueError(f"invalid --rows value {value!r}") from error
            if row <= 0:
                raise ValueError(f"row counts must be positive, got {row}")
            if row not in rows:
                rows.append(row)
    if not rows:
        raise ValueError("--rows did not contain a row count")
    return tuple(rows)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    data = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def _json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _manifest_path_for_key(cache: Path, key: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{64}", key):
        raise ValueError(f"cache key is not SHA-256: {key!r}")
    return cache / key[:2] / f"{key}.json"


def _manifest_for_spec(
    cache: Path,
    spec_hash: str,
    key: str | None,
) -> tuple[Path, dict[str, Any]]:
    cache = cache.resolve()
    if key is not None:
        paths = [_manifest_path_for_key(cache, key)]
    else:
        paths = sorted(cache.rglob("*.json"))
    matches: list[tuple[Path, dict[str, Any]]] = []
    for path in paths:
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            manifest.get("schema") == "b12x.cute.compile_manifest.v3"
            and manifest.get("compile_spec_hash") == spec_hash
            and manifest.get("object_sha256")
        ):
            matches.append((path, manifest))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one manifest for spec {spec_hash} in {cache}, "
            f"found {len(matches)}"
        )
    path, manifest = matches[0]
    manifest_key = str(manifest.get("cache_key", ""))
    if path != _manifest_path_for_key(cache, manifest_key):
        raise RuntimeError(f"manifest path/cache-key mismatch: {path}")
    if key is not None and manifest_key != key:
        raise RuntimeError(f"requested cache key {key} but manifest has {manifest_key}")
    return path, manifest


def _spec_facts(manifest: dict[str, Any]) -> dict[str, object]:
    raw = json.loads(str(manifest["compile_spec_json"]))
    if raw.get("kernel") != "attention.mla.sm120.prefill_mg":
        raise RuntimeError(f"unexpected kernel in compile spec: {raw.get('kernel')!r}")
    if raw.get("version") != 3:
        raise RuntimeError(f"unexpected compile-spec version: {raw.get('version')!r}")
    facts = raw.get("facts")
    if not isinstance(facts, list):
        raise RuntimeError("compile spec facts are not a list")
    return {
        str(fact[0]): fact[1]
        for fact in facts
        if isinstance(fact, list) and len(fact) == 2
    }


def _validate_spec_contract(
    manifest: dict[str, Any],
    case: PrefillSpec,
) -> None:
    facts = _spec_facts(manifest)
    expected = {
        "model_type": case.model_type,
        "compute_mode": case.compute_mode,
        "scale_format": case.scale_format,
        "num_heads": case.heads,
        "heads_per_cta": case.heads_per_cta or case.heads,
        "mg_n_hg": case.n_hg,
        "valid_hpb": case.valid_hpb,
        "pack_hilo_rows": case.pack_hilo_rows,
        "num_tiles": (case.topk + case.extra_topk) // 64,
        "page_block_size": 64,
        "topk_bucket": case.topk,
        "has_sink": int(case.has_sink),
    }
    if case.extra_topk:
        expected.update(
            {
                "has_extra": 1,
                "extra_topk_bucket": case.extra_topk,
                "num_main_tiles": case.topk // 64,
                "pbs_extra": 64,
            }
        )
    mismatches = {
        name: {"expected": expected_value, "actual": facts.get(name)}
        for name, expected_value in expected.items()
        if facts.get(name) != expected_value
    }
    if mismatches:
        raise RuntimeError(f"compile spec does not match {case.name}: {mismatches}")


def _load_exact(
    cache: Path,
    spec_hash: str,
    key: str | None,
    case: PrefillSpec,
) -> tuple[Any, dict[str, Any]]:
    cache = cache.resolve()
    manifest_path, manifest = _manifest_for_spec(cache, spec_hash, key)
    _validate_spec_contract(manifest, case)
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


def _verify_artifact(provenance: dict[str, Any]) -> dict[str, object]:
    manifest_path = Path(str(provenance["manifest_path"]))
    object_path = Path(str(provenance["object_path"]))
    observed = {
        "manifest_sha256": _sha256_file(manifest_path),
        "object_sha256": _sha256_file(object_path),
        "object_bytes": object_path.stat().st_size,
    }
    expected = {name: provenance[name] for name in observed}
    if observed != expected:
        raise RuntimeError(
            f"cached artifact changed during benchmark: expected={expected}, "
            f"observed={observed}"
        )
    return {"passed": True, **observed}


def _graph_topology(graph: torch.cuda.CUDAGraph) -> dict[str, object]:
    raw_graph = graph.raw_cuda_graph()
    error, _, node_count = cuda_driver.cuGraphGetNodes(raw_graph)
    if error != cuda_driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuGraphGetNodes count failed: {error}")
    error, nodes, returned_nodes = cuda_driver.cuGraphGetNodes(raw_graph, node_count)
    if error != cuda_driver.CUresult.CUDA_SUCCESS or returned_nodes != node_count:
        raise RuntimeError(
            f"cuGraphGetNodes failed: {error}, {returned_nodes}/{node_count}"
        )
    node_indices = {int(node): index for index, node in enumerate(nodes)}
    node_metadata: list[dict[str, object]] = []
    for index, node in enumerate(nodes):
        error, node_type = cuda_driver.cuGraphNodeGetType(node)
        if error != cuda_driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuGraphNodeGetType failed: {error}")
        metadata: dict[str, object] = {"index": index, "type": node_type.name}
        if node_type == cuda_driver.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL:
            error, params = cuda_driver.cuGraphKernelNodeGetParams(node)
            if error != cuda_driver.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"cuGraphKernelNodeGetParams failed: {error}")
            error, name = cuda_driver.cuFuncGetName(params.func)
            if error != cuda_driver.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"cuFuncGetName failed: {error}")
            metadata.update(
                {
                    "kernel_name": name.decode(),
                    "grid": [params.gridDimX, params.gridDimY, params.gridDimZ],
                    "block": [params.blockDimX, params.blockDimY, params.blockDimZ],
                    "dynamic_smem_bytes": params.sharedMemBytes,
                }
            )
        node_metadata.append(metadata)
    error, _, _, _, edge_count = cuda_driver.cuGraphGetEdges(raw_graph)
    if error != cuda_driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuGraphGetEdges count failed: {error}")
    error, sources, destinations, _, returned_edges = cuda_driver.cuGraphGetEdges(
        raw_graph, edge_count
    )
    if error != cuda_driver.CUresult.CUDA_SUCCESS or returned_edges != edge_count:
        raise RuntimeError(
            f"cuGraphGetEdges failed: {error}, {returned_edges}/{edge_count}"
        )
    return {
        "node_count": node_count,
        "kernel_node_count": sum(
            node["type"] == "CU_GRAPH_NODE_TYPE_KERNEL" for node in node_metadata
        ),
        "nodes": node_metadata,
        "edges": [
            [node_indices[int(source)], node_indices[int(destination)]]
            for source, destination in zip(sources, destinations, strict=True)
        ],
    }


def _correctness_metrics(
    got: torch.Tensor,
    expected: torch.Tensor,
) -> dict[str, object]:
    got_fp32 = got.float()
    expected_fp32 = expected.float()
    difference = got_fp32 - expected_fp32
    return {
        "finite": bool(torch.isfinite(got_fp32).all().item()),
        "nonzero": int(torch.count_nonzero(got_fp32).item()),
        "max_abs": float(difference.abs().max().item()),
        "relative_l2": float(
            torch.linalg.vector_norm(difference).item()
            / max(torch.linalg.vector_norm(expected_fp32).item(), 1.0e-30)
        ),
        "cosine": float(
            torch.nn.functional.cosine_similarity(
                got_fp32.flatten(), expected_fp32.flatten(), dim=0
            ).item()
        ),
        "sha256": _tensor_sha256(got),
    }


def _assert_live_topk_contract(
    *,
    live_indices: torch.Tensor,
    live_lengths: torch.Tensor,
    expected_indices: torch.Tensor,
    expected_lengths: torch.Tensor,
    rows: int,
    topk: int,
) -> None:
    if live_indices.shape != (rows, topk):
        raise AssertionError(
            f"unexpected live index shape: {tuple(live_indices.shape)}"
        )
    if live_lengths.shape != (rows,):
        raise AssertionError(
            f"unexpected live length shape: {tuple(live_lengths.shape)}"
        )
    if not torch.equal(live_indices, expected_indices):
        raise AssertionError("live indices do not match the installed scenario")
    if not torch.equal(live_lengths, expected_lengths):
        raise AssertionError("live lengths do not match the installed scenario")
    for row in range(rows):
        length = int(live_lengths[row].item())
        if not 0 < length <= topk:
            raise AssertionError(f"row {row}: invalid top-k length {length}")
        active = live_indices[row, :length]
        if not bool((active >= 0).all()) or not bool((active < topk).all()):
            raise AssertionError(f"row {row}: invalid active top-k index")
        if length < topk and int(live_indices[row, length].item()) != -1:
            raise AssertionError(f"row {row}: inactive tail is not poisoned")


def _run_rows(
    *,
    rows: int,
    case: PrefillSpec,
    labels: tuple[str, str],
    select_arm: Callable[[str], None],
    dispatch_counts: dict[str, int],
    precondition_replays: int,
    warmup_cycles: int,
    cycles: int,
    event_batch_cycles: int,
    replays_per_reported_sample: int,
    precondition_seconds: float,
    maximum_precondition_seconds: float,
    max_sm_clock_delta_mhz: float,
    l2_flush_bytes: int,
    expected_physical_gpu: int,
    device: torch.device,
) -> dict[str, object]:
    attn_sink: torch.Tensor | None = None
    extra_cache: torch.Tensor | None = None
    extra_indices: torch.Tensor | None = None
    extra_lengths: torch.Tensor | None = None
    extra_index_scenarios: tuple[torch.Tensor, torch.Tensor] | None = None
    extra_length_scenarios: tuple[torch.Tensor, torch.Tensor] | None = None
    scale_format: int | None = None
    immutable_tensors: dict[str, torch.Tensor]

    if case.family == "dsv4":
        inputs = _make_inputs(
            rows=rows,
            heads=case.heads,
            main_width=case.topk,
            extra_width=case.extra_topk,
            per_token=True,
            device=device,
        )
        if inputs.main_lengths is None:
            raise AssertionError("prefill benchmark requires per-token lengths")
        _poison_inactive_topk_tails(
            inputs.main_index_scenarios,
            inputs.main_length_scenarios,
        )
        if case.extra_topk:
            if (
                inputs.extra_cache is None
                or inputs.extra_indices is None
                or inputs.extra_index_scenarios is None
                or inputs.extra_lengths is None
                or inputs.extra_length_scenarios is None
            ):
                raise AssertionError("dual-cache prefill inputs are incomplete")
            _poison_inactive_topk_tails(
                inputs.extra_index_scenarios,
                inputs.extra_length_scenarios,
            )
        attn_sink = (
            torch.linspace(
                -0.8,
                0.6,
                case.heads,
                dtype=torch.float32,
                device=device,
            )
            if case.has_sink
            else None
        )
        expected = tuple(
            _reference(inputs, scenario, attn_sink=attn_sink) for scenario in range(2)
        )
        expected_lse = tuple(result[1] / math.log(2.0) for result in expected)
        live_q = inputs.q
        kv_cache = inputs.main_cache
        live_indices = inputs.main_indices
        live_lengths = inputs.main_lengths
        q_scenarios = inputs.q_scenarios
        index_scenarios = inputs.main_index_scenarios
        length_scenarios = inputs.main_length_scenarios
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
    elif case.family == "glm":
        if case.has_sink or case.extra_topk:
            raise AssertionError("GLM BF16-cache case cannot carry DSV4 extras")
        inputs_glm = _make_glm_inputs(
            rows=rows,
            heads=case.heads,
            width=case.topk,
            per_token=True,
            device=device,
        )
        if inputs_glm.lengths is None:
            raise AssertionError("GLM prefill benchmark requires per-token lengths")
        _poison_inactive_topk_tails(
            inputs_glm.index_scenarios,
            inputs_glm.length_scenarios,
        )
        expected = tuple(_glm_reference(inputs_glm, scenario) for scenario in range(2))
        expected_lse = tuple(result[1] for result in expected)
        live_q = inputs_glm.q
        kv_cache = inputs_glm.launch_cache
        live_indices = inputs_glm.indices
        live_lengths = inputs_glm.lengths
        q_scenarios = inputs_glm.q_scenarios
        index_scenarios = inputs_glm.index_scenarios
        length_scenarios = inputs_glm.length_scenarios
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
    else:
        if case.family != "glm-nvfp4" or case.has_sink or case.extra_topk:
            raise AssertionError(f"invalid prefill case family {case.family!r}")
        inputs_nvfp4 = _make_nvfp4_glm_inputs(
            rows=rows,
            heads=case.heads,
            width=case.topk,
            device=device,
        )
        _poison_inactive_topk_tails(
            inputs_nvfp4.index_scenarios,
            inputs_nvfp4.length_scenarios,
        )
        expected = tuple(
            _nvfp4_glm_reference(inputs_nvfp4, scenario) for scenario in range(2)
        )
        expected_lse = tuple(result[1] for result in expected)
        live_q = inputs_nvfp4.q
        kv_cache = inputs_nvfp4.launch_cache
        live_indices = inputs_nvfp4.indices
        live_lengths = inputs_nvfp4.lengths
        q_scenarios = inputs_nvfp4.q_scenarios
        index_scenarios = inputs_nvfp4.index_scenarios
        length_scenarios = inputs_nvfp4.length_scenarios
        sm_scale = _GLM_SM_SCALE
        scale_format = int(ScaleFormat.NVFP4_E4M3)

        def install(scenario: int) -> None:
            _install_nvfp4_glm_scenario(inputs_nvfp4, scenario)

        immutable_tensors = {
            "kv_cache": kv_cache,
            "dequant_nope": inputs_nvfp4.dequant_nope,
            "rope": inputs_nvfp4.rope,
            "q_scenario_0": q_scenarios[0],
            "q_scenario_1": q_scenarios[1],
            "indices_scenario_0": index_scenarios[0],
            "indices_scenario_1": index_scenarios[1],
            "lengths_scenario_0": length_scenarios[0],
            "lengths_scenario_1": length_scenarios[1],
        }

    if torch.allclose(expected[0][0], expected[1][0]):
        raise AssertionError("oracle scenarios are not distinct")

    output = torch.empty(
        (rows, case.heads, _GLM_V_DIM),
        dtype=torch.bfloat16,
        device=device,
    )
    lse_base2 = torch.empty(
        (rows, case.heads),
        dtype=torch.float32,
        device=device,
    )
    # MG prefill owns no split/merge scratch.  Keep an explicit caller-owned
    # sentinel so the symmetric serving contract still has a fixed workspace.
    fixed_workspace = torch.empty((1,), dtype=torch.uint8, device=device)
    stable_tensors = {
        "q": live_q,
        "kv_cache": kv_cache,
        "indices": live_indices,
        "lengths": live_lengths,
        "output": output,
        "lse": lse_base2,
        "workspace": fixed_workspace,
    }
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
    if attn_sink is not None:
        stable_tensors["attn_sink"] = attn_sink
    stable_pointers = {
        name: tensor.data_ptr() for name, tensor in stable_tensors.items()
    }
    if attn_sink is not None:
        immutable_tensors["attn_sink"] = attn_sink
    immutable_before = {
        name: _tensor_sha256(tensor) for name, tensor in immutable_tensors.items()
    }

    def launch_exact(label: str) -> tuple[torch.Tensor, torch.Tensor]:
        select_arm(label)
        return run_unified_prefill(
            q=live_q,
            kv_cache=kv_cache,
            topk_indices=live_indices,
            topk_length=live_lengths,
            sm_scale=sm_scale,
            page_block_size=_PAGE_SIZE,
            attn_sink=attn_sink,
            output=output,
            lse_out=lse_base2,
            workspace=fixed_workspace,
            scale_format=scale_format,
            extra_kv_cache=extra_cache,
            extra_indices=extra_indices,
            extra_topk_length=extra_lengths,
            extra_page_block_size=_PAGE_SIZE if extra_cache is not None else None,
        )

    def assert_stable_pointers() -> None:
        observed = {name: tensor.data_ptr() for name, tensor in stable_tensors.items()}
        if observed != stable_pointers:
            raise AssertionError(
                f"stable tensor pointer changed: {stable_pointers} -> {observed}"
            )

    stream = torch.cuda.Stream(device=device)
    graphs: dict[str, torch.cuda.CUDAGraph] = {}
    topologies: dict[str, dict[str, object]] = {}
    dispatch_before = dict(dispatch_counts)

    with torch.cuda.stream(stream):
        install(0)
    stream.synchronize()
    for label in labels:
        with torch.cuda.stream(stream):
            warm_output, warm_lse = launch_exact(label)
        stream.synchronize()
        if (
            warm_output.data_ptr() != output.data_ptr()
            or warm_lse.data_ptr() != lse_base2.data_ptr()
        ):
            raise AssertionError(f"{label}: launcher replaced caller-owned outputs")
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph, stream=stream):
            launch_exact(label)
        stream.synchronize()
        graphs[label] = graph
        topologies[label] = _graph_topology(graph)

    if topologies[labels[0]] != topologies[labels[1]]:
        raise AssertionError("arm graph topologies differ")
    topology = topologies[labels[0]]
    if topology["node_count"] != 1 or topology["kernel_node_count"] != 1:
        raise AssertionError(f"unexpected graph topology: {topology}")

    correctness: dict[str, dict[str, object]] = {}

    def validate(stage: str, scenario: int) -> None:
        with torch.cuda.stream(stream):
            install(scenario)
        stream.synchronize()
        _assert_live_topk_contract(
            live_indices=live_indices,
            live_lengths=live_lengths,
            expected_indices=index_scenarios[scenario],
            expected_lengths=length_scenarios[scenario],
            rows=rows,
            topk=case.topk,
        )
        if extra_indices is not None:
            assert extra_lengths is not None
            assert extra_index_scenarios is not None
            assert extra_length_scenarios is not None
            _assert_live_topk_contract(
                live_indices=extra_indices,
                live_lengths=extra_lengths,
                expected_indices=extra_index_scenarios[scenario],
                expected_lengths=extra_length_scenarios[scenario],
                rows=rows,
                topk=case.extra_topk,
            )
        assert_stable_pointers()
        arm_outputs: dict[str, torch.Tensor] = {}
        arm_lse: dict[str, torch.Tensor] = {}
        stage_result: dict[str, object] = {"scenario": scenario, "arms": {}}
        for label in labels:
            with torch.cuda.stream(stream):
                output.fill_(float("nan"))
                lse_base2.fill_(float("nan"))
            stream.synchronize()
            allocator_before = _allocator_counters(device)
            with torch.cuda.stream(stream):
                graphs[label].replay()
            stream.synchronize()
            allocator_after = _allocator_counters(device)
            if allocator_after != allocator_before:
                raise AssertionError(
                    f"{label}: replay allocated: "
                    f"{allocator_before} -> {allocator_after}"
                )
            assert_stable_pointers()
            if torch.isnan(output).any() or torch.isnan(lse_base2).any():
                raise AssertionError(f"{label}: replay left poisoned output")
            _assert_output(
                output,
                expected[scenario][0],
                label=f"{label} {stage} rows={rows} scenario={scenario}",
            )
            _assert_prefill_boundary_heads(
                output,
                expected[scenario][0],
                n_hg=case.n_hg,
            )
            torch.testing.assert_close(
                lse_base2,
                expected_lse[scenario],
                atol=6.0e-2,
                rtol=2.0e-2,
            )
            if (
                not bool(torch.isfinite(lse_base2).all().item())
                or int(torch.count_nonzero(lse_base2).item()) == 0
            ):
                raise AssertionError(f"{label}: invalid LSE output")
            arm_outputs[label] = output.clone()
            arm_lse[label] = lse_base2.clone()
            stage_result["arms"][label] = {
                "output": _correctness_metrics(output, expected[scenario][0]),
                "lse_max_abs": float(
                    (lse_base2 - expected_lse[scenario]).abs().max().item()
                ),
                "lse_sha256": _tensor_sha256(lse_base2),
                "allocator_before": allocator_before,
                "allocator_after": allocator_after,
                "passed": True,
            }
        output_equal = torch.equal(arm_outputs[labels[0]], arm_outputs[labels[1]])
        lse_equal = torch.equal(arm_lse[labels[0]], arm_lse[labels[1]])
        if not output_equal or not lse_equal:
            raise AssertionError(
                f"{stage}: arms are not bit exact "
                f"(output={output_equal}, lse={lse_equal})"
            )
        stage_result.update(
            {
                "output_arms_bit_exact": True,
                "lse_arms_bit_exact": True,
                "passed": True,
            }
        )
        correctness[stage] = stage_result

    validate("pre", 0)
    timed_live_tensors = {
        "q": live_q,
        "indices": live_indices,
        "lengths": live_lengths,
    }
    if extra_indices is not None:
        assert extra_lengths is not None
        timed_live_tensors.update(
            {
                "extra_indices": extra_indices,
                "extra_lengths": extra_lengths,
            }
        )
    timed_live_before = {
        name: _tensor_sha256(tensor) for name, tensor in timed_live_tensors.items()
    }

    # Populate the fixed cold-L2 buffer cache before the enclosing allocator
    # snapshot. The shared timer separately verifies each condition around the
    # exact sample interval.
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
            "CUDA allocator state changed across MLA prefill graph timing: "
            f"{allocator_before_timing} -> {allocator_after_timing}"
        )
    compile_spec_hashes = {label: case.spec_hash for label in labels}
    for condition in conditions.values():
        condition["compile_spec_hashes"] = compile_spec_hashes
        condition["all_graph_spec_hashes"] = [case.spec_hash]

    timed_live_after = {
        name: _tensor_sha256(tensor) for name, tensor in timed_live_tensors.items()
    }
    if timed_live_after != timed_live_before:
        raise AssertionError("live timing input changed during replay")
    validate("post", 1)
    assert_stable_pointers()
    immutable_after = {
        name: _tensor_sha256(tensor) for name, tensor in immutable_tensors.items()
    }
    if immutable_after != immutable_before:
        raise AssertionError("read-only benchmark input changed")
    dispatch_after = dict(dispatch_counts)

    return {
        "rows": rows,
        "shape": {
            "family": case.family,
            "heads": case.heads,
            "heads_per_cta": case.heads_per_cta or case.heads,
            "valid_hpb": case.valid_hpb,
            "pack_hilo_rows": case.pack_hilo_rows,
            "topk": case.topk,
            "extra_topk": case.extra_topk,
            "mg_n_hg": case.n_hg,
            "compute_mode": "fp8" if case.compute_mode == 0 else "bf16",
            "scale_format": case.scale_format,
            "has_sink": case.has_sink,
        },
        "workspace": {
            "single_pass_scratch_bytes": 0,
            "caller_owned_sentinel_bytes": fixed_workspace.numel(),
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
                label: dispatch_after[label] - dispatch_before[label]
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
        "timing": {
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
        },
    }


def main() -> None:
    args = _args()
    rows = _row_sweep(args.rows)
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

    case = _SPECS[args.spec_hash]
    labels = (args.a_label, args.b_label)
    compiled_a, provenance_a = _load_exact(
        args.a_cache,
        args.spec_hash,
        args.a_key,
        case,
    )
    compiled_b, provenance_b = _load_exact(
        args.b_cache,
        args.spec_hash,
        args.b_key,
        case,
    )
    compiled = {labels[0]: compiled_a, labels[1]: compiled_b}
    provenance = {labels[0]: provenance_a, labels[1]: provenance_b}
    if provenance_a["compile_spec_json"] != provenance_b["compile_spec_json"]:
        raise RuntimeError("A/B compile specs differ")
    if provenance_a["kernel_id"] != provenance_b["kernel_id"]:
        raise RuntimeError("A/B kernel IDs differ")

    integrity_initial = {
        label: _verify_artifact(record) for label, record in provenance.items()
    }
    l2_flush_bytes = resolve_l2_flush_bytes(args.l2_flush_bytes)
    if make_l2_flush_fn(True, l2_flush_bytes) is None:
        raise AssertionError("cold-L2 flush function was not constructed")
    torch.cuda.synchronize(device)
    gpu_mode_initial = gpu_mode_snapshot(args.expected_physical_gpu)

    active_label: list[str | None] = [None]
    dispatch_counts = {label: 0 for label in labels}
    observed_specs: set[str] = set()
    original_launch = prefill_mg.b12x_launch
    previous_gate = os.environ.get(_MG_GATE_ENV)
    os.environ[_MG_GATE_ENV] = "1"

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
        observed_specs.add(compile_spec.hash_key)
        if compile_spec.hash_key != case.spec_hash:
            raise RuntimeError(
                f"launcher produced unexpected spec {compile_spec.hash_key}"
            )
        dispatch_counts[label] += 1
        return cute_compiler.run_compiled(compiled[label], runtime_args)

    prefill_mg.b12x_launch = exact_dispatch
    row_results: list[dict[str, object]] = []
    integrity_by_rows: dict[str, dict[str, dict[str, object]]] = {}
    try:
        for row_count in rows:
            integrity_before = {
                label: _verify_artifact(record) for label, record in provenance.items()
            }

            def select_arm(label: str) -> None:
                active_label[0] = label

            row_result = _run_rows(
                rows=row_count,
                case=case,
                labels=labels,
                select_arm=select_arm,
                dispatch_counts=dispatch_counts,
                precondition_replays=args.precondition_replays,
                warmup_cycles=args.warmup_cycles,
                cycles=args.cycles,
                event_batch_cycles=args.event_batch_cycles,
                replays_per_reported_sample=args.replays_per_reported_sample,
                precondition_seconds=args.precondition_seconds,
                maximum_precondition_seconds=args.maximum_precondition_seconds,
                max_sm_clock_delta_mhz=args.max_sm_clock_delta_mhz,
                l2_flush_bytes=l2_flush_bytes,
                expected_physical_gpu=args.expected_physical_gpu,
                device=device,
            )
            integrity_after = {
                label: _verify_artifact(record) for label, record in provenance.items()
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
        prefill_mg.b12x_launch = original_launch
        active_label[0] = None
        if previous_gate is None:
            os.environ.pop(_MG_GATE_ENV, None)
        else:
            os.environ[_MG_GATE_ENV] = previous_gate

    if observed_specs != {case.spec_hash}:
        raise AssertionError(f"unexpected observed specs: {observed_specs}")
    integrity_final = {
        label: _verify_artifact(record) for label, record in provenance.items()
    }
    if integrity_final != integrity_initial:
        raise RuntimeError("artifact integrity changed during benchmark")
    gpu_mode_final = gpu_mode_snapshot(args.expected_physical_gpu)
    result: dict[str, object] = {
        "schema": "b12x.attention.mla.prefill_mg.exact_cache_abba.v4",
        "evidence_status": args.evidence_status,
        "command": [sys.executable, *sys.argv],
        "case": {
            "name": case.name,
            "compile_spec_hash": case.spec_hash,
            "family": case.family,
            "heads": case.heads,
            "heads_per_cta": case.heads_per_cta or case.heads,
            "valid_hpb": case.valid_hpb,
            "pack_hilo_rows": case.pack_hilo_rows,
            "topk": case.topk,
            "extra_topk": case.extra_topk,
            "mg_n_hg": case.n_hg,
            "compute_mode": "fp8" if case.compute_mode == 0 else "bf16",
            "scale_format": case.scale_format,
            "has_sink": case.has_sink,
            "rows": list(rows),
        },
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
            "mg_gate_override": {_MG_GATE_ENV: "1"},
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
                "prefill_dispatch": _sha256_file(
                    REPO_ROOT / "b12x/attention/mla/prefill.py"
                ),
                "prefill_mg": _sha256_file(
                    REPO_ROOT / "b12x/attention/mla/prefill_mg.py"
                ),
                "compiler": _sha256_file(REPO_ROOT / "b12x/cute/compiler.py"),
                "corpus_helpers": _sha256_file(
                    REPO_ROOT / "tests/test_attention_mla_unified_corpus.py"
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
