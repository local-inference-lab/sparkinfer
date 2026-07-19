#!/usr/bin/env python3
"""Exact-object warm/cold CUDA-graph ABBA for TP-MoE dynamic kernels.

This benchmark closes the two frozen-v5 CUTLASS 4.6 TP-MoE resource
exceptions through the production planned/bound serving entrypoint:

* NVFP4 dynamic prefill, M128/E4/K512/N128/top-2;
* materialized W4A8-MX prefill, M4096/E16/K4096/N1024/top-4.

The compile contract is read from each cache manifest and reconstructed from
the live production dispatch arguments.  A mismatch aborts before capture.
Both exact objects then use one binding, one caller-owned scratch allocation,
one output allocation, and identical live-input addresses.  The benchmark
requires GPU-reference correctness before timing.  The production default uses
atomic output reduction, so exact-object arm equality is checked against a
multi-replay, elementwise BF16 envelope derived independently for each arm;
deterministic specializations still require strict bit equality.  The benchmark
mutates every live input in place and repeats those gates after balanced warm-
and cold-L2 CUDA-graph ABBA timing.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, fields, is_dataclass
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import torch
from cuda.bindings import driver as cuda_driver

import b12x.cute.compiler as cute_compiler
import b12x.integration.tp_moe as tp_moe
from benchmarks.common import (
    make_l2_flush_fn,
    nvidia_smi_gpu_mode_snapshot,
)
from validation.cutlass_migration.core.evidence_status import (
    add_evidence_status_argument,
)
from validation.cutlass_migration.core.exact_cache_abba import (
    allocator_counters,
    time_conditions,
)
from validation.cutlass_migration.paths import REPO_ROOT
from b12x.moe.fused.reference import compare_to_reference


@dataclass(frozen=True)
class DynamicCase:
    name: str
    spec_hash: str
    corpus_nodeid: str
    m: int
    min_cosine: float
    max_normalized_rmse: float


_CASES = {
    case.name: case
    for case in (
        DynamicCase(
            name="nvfp4-prefill-m128",
            spec_hash=(
                "0bc508dbd1653bc3e566d299b82587caa85c1c5c1c320e70edf24831724f9d02"
            ),
            corpus_nodeid=(
                "tests/test_cute_migration_moe_standard_corpus.py::"
                "test_standard_moe_dynamic_prefill_live_graph_oracle"
            ),
            m=128,
            min_cosine=0.999,
            max_normalized_rmse=0.03,
        ),
        DynamicCase(
            name="w4a8-mx-materialized-m4096",
            spec_hash=(
                "155c03993f48e837eff1d8c8016fb88acdd1e7e940e0634140a3fdcad9a5ef27"
            ),
            corpus_nodeid=(
                "tests/test_w4a8_migration_corpus.py::"
                "test_w4a8_materialized_routing_phase1_phase2_matches_oracle_under_graph"
            ),
            m=4096,
            min_cosine=0.999,
            max_normalized_rmse=0.03,
        ),
    )
}

_FROZEN_SOURCE_MANIFEST_SHA256 = (
    "77c74780c843ac2367a6591ef7099ab89f0f58d6456459b65a5a7ae92cda302e"
)
_FROZEN_RELEVANT_SOURCE_SHA256 = {
    "integration_tp_moe": (
        "0cf496b5dd4091782064408180e8ffd03cff14af935ef7b1d32f940b7374cf2f"
    ),
    "dynamic_kernel": (
        "7b0ec4c820cbbd01a49997a70304ab3dfb687ec07e985817a6ec0ccb3bff84ea"
    ),
}


@dataclass
class CaseState:
    binding: object
    output: torch.Tensor
    scratch_plan: object
    scratch: tuple[torch.Tensor, ...]
    live_a: torch.Tensor
    live_topk_ids: torch.Tensor
    live_topk_weights: torch.Tensor
    scenarios: tuple[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]
    references: tuple[torch.Tensor, torch.Tensor]
    owners: dict[str, object]
    immutable_roots: dict[str, object]


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_evidence_status_argument(parser)
    parser.add_argument("--a-cache", type=Path, required=True)
    parser.add_argument("--a-key", action="append", default=[])
    parser.add_argument("--a-label", default="cutlass-4.5.2")
    parser.add_argument("--a-cutlass-version", default="4.5.2")
    parser.add_argument("--b-cache", type=Path, required=True)
    parser.add_argument("--b-key", action="append", default=[])
    parser.add_argument("--b-label", default="cutlass-4.6.0")
    parser.add_argument("--b-cutlass-version", default="4.6.0")
    parser.add_argument(
        "--case",
        action="append",
        choices=tuple(_CASES),
        default=[],
        help="case to run; repeat as needed (default: both frozen-v5 cases)",
    )
    parser.add_argument("--precondition-cycles", type=int, default=250)
    parser.add_argument("--precondition-seconds", type=float, default=5.0)
    parser.add_argument("--maximum-precondition-seconds", type=float, default=60.0)
    parser.add_argument("--max-sm-clock-delta-mhz", type=float, default=60.0)
    parser.add_argument("--warmup-cycles", type=int, default=100)
    parser.add_argument("--cycles", type=int, default=600)
    parser.add_argument("--event-batch-cycles", type=int, default=25)
    parser.add_argument("--replays-per-reported-sample", type=int, default=1)
    parser.add_argument("--l2-flush-bytes", type=int, default=0)
    parser.add_argument(
        "--expected-physical-gpu",
        type=int,
        choices=(4, 5),
        required=True,
        help="required physical nvidia-smi index; rs-18 is restricted to GPUs 4/5",
    )
    parser.add_argument(
        "--expected-current-package-fingerprint",
        required=True,
        help=(
            "required b12x fingerprint for the current host source; frozen object "
            "fingerprints are validated and reported separately"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    digest = hashlib.sha256()
    digest.update(f"{tuple(tensor.shape)}:{tensor.dtype}".encode())
    digest.update(
        tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    )
    return digest.hexdigest()


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "missing"


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


def _key_by_case(raw: list[str], selected: tuple[str, ...]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in raw:
        if "=" not in item:
            if len(selected) != 1:
                raise ValueError(
                    "bare --a-key/--b-key is valid only with one selected case; "
                    "otherwise use CASE=KEY"
                )
            case_name, key = selected[0], item
        else:
            case_name, key = item.split("=", 1)
        if case_name not in selected:
            raise ValueError(f"cache key names unselected case {case_name!r}")
        if case_name in result:
            raise ValueError(f"duplicate cache key for case {case_name!r}")
        if not re.fullmatch(r"[0-9a-f]{64}", key):
            raise ValueError(f"cache key is not SHA-256: {key!r}")
        result[case_name] = key
    return result


def _manifest_for_spec(
    cache: Path,
    spec_hash: str,
    key: str | None,
) -> tuple[Path, dict[str, Any]]:
    cache = cache.resolve()
    paths = (
        [_manifest_path_for_key(cache, key)] if key else sorted(cache.rglob("*.json"))
    )
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
    expected_path = _manifest_path_for_key(cache, manifest_key)
    if path.resolve() != expected_path.resolve():
        raise RuntimeError(f"manifest path/cache-key mismatch: {path}")
    if key is not None and manifest_key != key:
        raise RuntimeError(f"requested cache key {key} but manifest has {manifest_key}")
    return path, manifest


def _toolchain_version(manifest: Mapping[str, Any], package: str) -> str:
    versions = {
        str(item[0]): str(item[1])
        for item in manifest.get("toolchain", [])
        if isinstance(item, list) and len(item) >= 2
    }
    return versions.get(package, "missing")


def _load_exact(
    cache: Path,
    case: DynamicCase,
    key: str | None,
    expected_cutlass_version: str,
) -> tuple[Any, dict[str, Any]]:
    cache = cache.resolve()
    manifest_path, manifest = _manifest_for_spec(cache, case.spec_hash, key)
    observed_cutlass = _toolchain_version(manifest, "cutlass_dsl")
    if observed_cutlass != expected_cutlass_version:
        raise RuntimeError(
            f"{case.name}: expected CUTLASS DSL {expected_cutlass_version}, "
            f"manifest reports {observed_cutlass}"
        )
    cache_key = str(manifest["cache_key"])
    object_path = manifest_path.with_suffix(".o")
    object_sha256 = _sha256_file(object_path)
    if object_sha256 != manifest["object_sha256"]:
        raise RuntimeError(f"object digest mismatch: {object_path}")
    if object_path.stat().st_size != int(manifest["object_bytes"]):
        raise RuntimeError(f"object byte-count mismatch: {object_path}")
    ptx_evidence_path = manifest_path.with_suffix(".ptx.json")
    ptx_evidence = json.loads(ptx_evidence_path.read_text(encoding="utf-8"))
    ptx_object = ptx_evidence.get("object")
    if (
        ptx_evidence.get("schema") != "b12x.cute.frontend_ptx.v3"
        or ptx_evidence.get("cache_key") != cache_key
        or ptx_evidence.get("compile_spec_hash") != case.spec_hash
        or not isinstance(ptx_object, dict)
        or ptx_object.get("sha256") != object_sha256
        or int(ptx_object.get("bytes", -1)) != object_path.stat().st_size
    ):
        raise RuntimeError(
            f"PTX evidence does not describe the exact cached object: "
            f"{ptx_evidence_path}"
        )

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
        "ptx_evidence_path": str(ptx_evidence_path),
        "ptx_evidence_sha256": _sha256_file(ptx_evidence_path),
        "source_ptxas": ptx_evidence.get("source_ptxas"),
        "embedded_cubin": {
            key: ptx_object.get(key)
            for key in (
                "embedded_cubin_bytes",
                "embedded_cubin_offset",
                "embedded_cubin_sha256",
            )
        },
        "compile_spec_hash": manifest["compile_spec_hash"],
        "compile_spec_json": manifest["compile_spec_json"],
        "compile_options": manifest.get("compile_options"),
        "compile_environment": manifest.get("compile_environment"),
        "semantic_key": manifest.get("semantic_key"),
        "package_fingerprint": manifest.get("package_fingerprint"),
        "target": manifest.get("target"),
        "kernel_id": manifest.get("kernel_id"),
        "launch_metadata": manifest.get("launch_metadata"),
        "toolchain": manifest.get("toolchain"),
        "cutlass_dsl_version": observed_cutlass,
    }


def _verify_artifact(provenance: Mapping[str, Any]) -> dict[str, object]:
    manifest_path = Path(str(provenance["manifest_path"]))
    object_path = Path(str(provenance["object_path"]))
    observed = {
        "manifest_sha256": _sha256_file(manifest_path),
        "object_sha256": _sha256_file(object_path),
        "object_bytes": object_path.stat().st_size,
        "ptx_evidence_sha256": _sha256_file(Path(str(provenance["ptx_evidence_path"]))),
    }
    expected = {name: provenance[name] for name in observed}
    if observed != expected:
        raise RuntimeError(
            f"cached artifact changed during benchmark: expected={expected}, "
            f"observed={observed}"
        )
    return {"passed": True, **observed}


def _legacy_compile_args(provenance: Mapping[str, Any]) -> list[object]:
    spec = json.loads(str(provenance["compile_spec_json"]))
    if spec.get("kernel") != "integration.tp_moe.dynamic" or spec.get("version") != 1:
        raise RuntimeError(f"unexpected TP-MoE compile spec: {spec}")
    facts = spec.get("facts")
    if (
        not isinstance(facts, list)
        or len(facts) != 4
        or facts[:3] != ["legacy", "integration.tp_moe.dynamic", 1]
        or not isinstance(facts[3], list)
        or len(facts[3]) != 21
    ):
        raise RuntimeError(f"unexpected legacy compile facts: {facts}")
    indexed: dict[int, object] = {}
    for field in facts[3]:
        if (
            not isinstance(field, list)
            or len(field) != 3
            or field[0] != "field"
            or not re.fullmatch(r"arg\d+", str(field[1]))
        ):
            raise RuntimeError(f"invalid legacy compile field: {field}")
        index = int(str(field[1])[3:])
        if index in indexed:
            raise RuntimeError(f"duplicate legacy compile field arg{index}")
        indexed[index] = field[2]
    expected_indices = set(range(21))
    if set(indexed) != expected_indices:
        raise RuntimeError(f"expected legacy args 0..20, got {sorted(indexed)}")
    return [indexed[index] for index in range(21)]


def _validate_frozen_case_contract(
    case: DynamicCase, compile_args: list[object]
) -> None:
    expected_by_case = {
        "nvfp4-prefill-m128": [
            "nvfp4",
            "dynamic",
            4,
            512,
            128,
            2,
            [64, 128],
            ["repr", "torch", "dtype", "torch.int32"],
            False,
            "silu",
            None,
            1.0,
            0.0,
            False,
            True,
            False,
            False,
            "materialized_queue",
            False,
            False,
            False,
        ],
        "w4a8-mx-materialized-m4096": [
            "w4a8_mx",
            "dynamic",
            16,
            4096,
            1024,
            4,
            [64, 128],
            ["repr", "torch", "dtype", "torch.int32"],
            True,
            "silu",
            None,
            1.0,
            0.0,
            False,
            True,
            False,
            False,
            "materialized_queue",
            True,
            False,
            True,
        ],
    }
    expected = expected_by_case[case.name]
    if compile_args != expected:
        mismatches = {
            f"arg{index}": {"expected": expected[index], "actual": actual}
            for index, actual in enumerate(compile_args)
            if actual != expected[index]
        }
        raise RuntimeError(
            f"{case.name}: manifest no longer matches frozen-v5 contract: {mismatches}"
        )


def _validate_pair(
    case: DynamicCase,
    provenance: Mapping[str, Mapping[str, Any]],
    labels: tuple[str, str],
) -> list[object]:
    a = provenance[labels[0]]
    b = provenance[labels[1]]
    for field in (
        "compile_spec_hash",
        "compile_spec_json",
        "package_fingerprint",
        "semantic_key",
        "kernel_id",
        "launch_metadata",
    ):
        if a[field] != b[field]:
            raise RuntimeError(f"{case.name}: arm {field} differs")
    if a["compile_spec_hash"] != case.spec_hash:
        raise RuntimeError(f"{case.name}: wrong compile-spec hash")
    if a["kernel_id"] != "integration.tp_moe.dynamic":
        raise RuntimeError(f"{case.name}: wrong kernel ID {a['kernel_id']!r}")
    expected_target = {
        "nvfp4-prefill-m128": "b12x.integration.tp_moe._DynamicMoELaunch",
        "w4a8-mx-materialized-m4096": ("b12x.integration.tp_moe._DynamicMoEW4A8Launch"),
    }[case.name]
    if a["target"] != expected_target or b["target"] != expected_target:
        raise RuntimeError(
            f"{case.name}: unexpected manifest targets {a['target']!r}, {b['target']!r}"
        )
    launch_metadata = a["launch_metadata"]
    if (
        not isinstance(launch_metadata, dict)
        or launch_metadata.get("status") != "exact"
        or not isinstance(launch_metadata.get("launch_dynamic_smem_bytes"), dict)
    ):
        raise RuntimeError(f"{case.name}: launch metadata is not exact")
    compile_args = _legacy_compile_args(a)
    _validate_frozen_case_contract(case, compile_args)
    return compile_args


@contextmanager
def _specialization_environment(compile_args: list[object]):
    tile = compile_args[6]
    if not isinstance(tile, list) or len(tile) != 2:
        raise RuntimeError(f"invalid tile compile fact: {tile}")
    controlled = {
        "B12X_DYNAMIC_TILE_MN": f"{int(tile[0])}x{int(tile[1])}",
        "B12X_DYNAMIC_WORK_SOURCE": str(compile_args[17]),
        "B12X_ENABLE_DYNAMIC_DOWN_SCALE": "1" if compile_args[13] else "0",
        "B12X_DYNAMIC_W4A8_SHARE_INPUT": "1" if compile_args[14] else "0",
        "B12X_DYNAMIC_SWAP_AB": "1" if compile_args[15] else "0",
        "B12X_DYNAMIC_DETERMINISTIC_OUTPUT": "1" if compile_args[16] else "0",
        "B12X_DYNAMIC_W4A8_MATERIALIZED": "1" if compile_args[20] else "0",
        "B12X_DYNAMIC_ENABLE_MULTICTA": "1",
    }
    previous = {name: os.environ.get(name) for name in controlled}
    previous_override = tp_moe._DYNAMIC_TILE_MN_OVERRIDE
    try:
        tp_moe._DYNAMIC_TILE_MN_OVERRIDE = None
        os.environ.update(controlled)
        tp_moe.clear_tp_moe_caches()
        yield controlled
    finally:
        tp_moe.clear_tp_moe_caches()
        tp_moe._DYNAMIC_TILE_MN_OVERRIDE = previous_override
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _allocate_binding(
    *,
    experts: object,
    a: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    output: torch.Tensor,
    quant_mode: str,
    fast_math: bool,
) -> tuple[object, tuple[torch.Tensor, ...], object]:
    from b12x.integration import TPMoEScratchCaps, plan_tp_moe_scratch

    scratch_plan = plan_tp_moe_scratch(
        TPMoEScratchCaps(
            max_tokens=int(a.shape[0]),
            num_topk=int(topk_ids.shape[1]),
            device=a.device,
            weight_plan=experts.plan,
            core_token_counts=(int(a.shape[0]),),
            route_num_experts=0,
            quant_mode=quant_mode,
            frozen=True,
        )
    )
    scratch = tuple(
        torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
        for spec in scratch_plan.scratch_specs()
    )
    binding = scratch_plan.bind(
        scratch=scratch,
        a=a,
        experts=experts,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        output=output,
        input_scales_static=True,
        fast_math=fast_math,
    )
    if not scratch_plan.caps.frozen:
        raise AssertionError("TP-MoE scratch plan is not frozen")
    if scratch_plan.launch_plan.implementation != "dynamic":
        raise AssertionError(
            f"expected dynamic scratch plan, got {scratch_plan.launch_plan.implementation}"
        )
    if binding.implementation != "dynamic" or binding.output is not output:
        raise AssertionError("binding does not own the requested dynamic output")
    return scratch_plan, scratch, binding


def _build_nvfp4_state(case: DynamicCase, compile_args: list[object]) -> CaseState:
    from tests.helpers import prepare_tp_moe_fp4_experts
    from tests.test_cute_migration_moe_standard_corpus import (
        _E,
        _K,
        _N,
        _TOPK,
        _make_inputs,
        _make_nvfp4_weights,
        _nvfp4_oracle,
    )

    if tuple(map(int, compile_args[2:6])) != (_E, _K, _N, _TOPK):
        raise AssertionError("standard-MoE corpus constants do not match manifest")
    device = torch.device("cuda")
    weights = _make_nvfp4_weights(device, seed=201)
    templates = (
        _make_inputs(device, m=case.m, seed=202, route_shift=0),
        _make_inputs(device, m=case.m, seed=203, route_shift=2),
    )
    references = tuple(_nvfp4_oracle(weights, inputs) for inputs in templates)
    live_a = templates[0].a.clone()
    live_topk_ids = templates[0].topk_ids.clone()
    live_topk_weights = templates[0].topk_weights.clone()
    experts = prepare_tp_moe_fp4_experts(
        a=live_a,
        a1_gscale=weights.a1_scale,
        w1_fp4=weights.w1_fp4,
        w1_blockscale=weights.w1_scale,
        w1_alphas=weights.w1_alpha,
        a2_gscale=weights.a2_scale,
        w2_fp4=weights.w2_fp4,
        w2_blockscale=weights.w2_scale,
        w2_alphas=weights.w2_alpha,
        activation="silu",
        quant_mode="nvfp4",
        source_format="modelopt_nvfp4",
        w13_layout="w13",
    )
    output = torch.empty_like(live_a)
    scratch_plan, scratch, binding = _allocate_binding(
        experts=experts,
        a=live_a,
        topk_ids=live_topk_ids,
        topk_weights=live_topk_weights,
        output=output,
        quant_mode="nvfp4",
        fast_math=bool(compile_args[8]),
    )
    return CaseState(
        binding=binding,
        output=output,
        scratch_plan=scratch_plan,
        scratch=scratch,
        live_a=live_a,
        live_topk_ids=live_topk_ids,
        live_topk_weights=live_topk_weights,
        scenarios=tuple(
            (inputs.a, inputs.topk_ids, inputs.topk_weights) for inputs in templates
        ),
        references=references,
        owners={"weights": weights, "experts": experts},
        immutable_roots={
            "weights": weights,
            "experts": experts,
            "scenario_templates": templates,
            "references": references,
        },
    )


def _build_w4a8_state(case: DynamicCase, compile_args: list[object]) -> CaseState:
    from b12x.moe.fused.reference import moe_reference_w4a8_mx
    from tests.test_w4a8_mx_tp_moe import (
        _E,
        _K,
        _N,
        _TOPK,
        _prepare,
        _routed_inputs,
        _weights,
    )

    if tuple(map(int, compile_args[2:6])) != (_E, _K, _N, _TOPK):
        raise AssertionError("W4A8 corpus constants do not match manifest")
    weights = _weights(seed=3_000)
    templates = (
        _routed_inputs(case.m, 3_001),
        _routed_inputs(case.m, 3_002),
    )
    references = tuple(
        moe_reference_w4a8_mx(
            a.float(),
            weights["w13_fp4"],
            weights["w13_mx"],
            None,
            weights["alphas"],
            weights["w2_fp4"],
            weights["w2_mx"],
            None,
            weights["alphas"],
            topk_ids,
            topk_weights,
            _E,
            _K,
            _N,
            activation="silu",
        )
        for a, topk_ids, topk_weights in templates
    )
    # Preparation transfers the checkpoint allocations in place.  Both
    # references above therefore have to exist before this call.
    experts = _prepare(weights)
    live_a = templates[0][0].clone()
    live_topk_ids = templates[0][1].clone()
    live_topk_weights = templates[0][2].clone()
    output = torch.empty_like(live_a)
    scratch_plan, scratch, binding = _allocate_binding(
        experts=experts,
        a=live_a,
        topk_ids=live_topk_ids,
        topk_weights=live_topk_weights,
        output=output,
        quant_mode="w4a8_mx",
        fast_math=bool(compile_args[8]),
    )
    return CaseState(
        binding=binding,
        output=output,
        scratch_plan=scratch_plan,
        scratch=scratch,
        live_a=live_a,
        live_topk_ids=live_topk_ids,
        live_topk_weights=live_topk_weights,
        scenarios=templates,
        references=references,
        owners={"checkpoint_storage": weights, "experts": experts},
        immutable_roots={
            "checkpoint_storage_after_transfer": weights,
            "experts": experts,
            "scenario_templates": templates,
            "references": references,
        },
    )


def _build_case_state(case: DynamicCase, compile_args: list[object]) -> CaseState:
    if case.name == "nvfp4-prefill-m128":
        return _build_nvfp4_state(case, compile_args)
    if case.name == "w4a8-mx-materialized-m4096":
        return _build_w4a8_state(case, compile_args)
    raise AssertionError(f"unsupported case {case.name!r}")


def _collect_tensor_leaves(
    value: object,
    *,
    prefix: str,
    result: dict[str, torch.Tensor],
    seen: set[int],
) -> None:
    if isinstance(value, torch.Tensor):
        result[prefix] = value
        return
    if value is None or isinstance(value, (str, bytes, int, float, bool, torch.dtype)):
        return
    object_id = id(value)
    if object_id in seen:
        return
    seen.add(object_id)
    if isinstance(value, Mapping):
        for key, child in value.items():
            _collect_tensor_leaves(
                child,
                prefix=f"{prefix}.{key}",
                result=result,
                seen=seen,
            )
    elif isinstance(value, Sequence):
        for index, child in enumerate(value):
            _collect_tensor_leaves(
                child,
                prefix=f"{prefix}.{index}",
                result=result,
                seen=seen,
            )
    elif is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            _collect_tensor_leaves(
                getattr(value, field.name),
                prefix=f"{prefix}.{field.name}",
                result=result,
                seen=seen,
            )


def _tensor_leaves(roots: Mapping[str, object]) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    seen: set[int] = set()
    for name, value in roots.items():
        _collect_tensor_leaves(value, prefix=name, result=result, seen=seen)
    return result


def _stable_tensors(state: CaseState) -> dict[str, torch.Tensor]:
    return _tensor_leaves(
        {
            "binding": state.binding,
            "scratch": state.scratch,
            "owners": state.owners,
            "output": state.output,
        }
    )


def _install_scenario(state: CaseState, scenario: int) -> None:
    a, topk_ids, topk_weights = state.scenarios[scenario]
    state.live_a.copy_(a)
    state.live_topk_ids.copy_(topk_ids)
    state.live_topk_weights.copy_(topk_weights)


def _assert_live_contract(state: CaseState, scenario: int, num_experts: int) -> None:
    a, topk_ids, topk_weights = state.scenarios[scenario]
    if not torch.equal(state.live_a, a):
        raise AssertionError("live activations do not match installed scenario")
    if not torch.equal(state.live_topk_ids, topk_ids):
        raise AssertionError("live top-k IDs do not match installed scenario")
    if not torch.equal(state.live_topk_weights, topk_weights):
        raise AssertionError("live top-k weights do not match installed scenario")
    ids = state.live_topk_ids
    weights = state.live_topk_weights
    if ids.dtype is not torch.int32 or not ids.is_contiguous():
        raise AssertionError("top-k IDs must be contiguous int32")
    if not bool(((ids >= 0) & (ids < num_experts)).all().item()):
        raise AssertionError("top-k IDs are out of range")
    if ids.shape[1] > 1:
        ordered, _ = torch.sort(ids, dim=1)
        if bool((ordered[:, 1:] == ordered[:, :-1]).any().item()):
            raise AssertionError("a token routes to the same expert more than once")
    if not bool((weights > 0).all().item()):
        raise AssertionError("top-k weights must be positive")
    torch.testing.assert_close(
        weights.sum(dim=1),
        torch.ones(weights.shape[0], dtype=weights.dtype, device=weights.device),
        rtol=0,
        atol=1.0e-6,
    )


def _assert_scenarios_distinct(state: CaseState) -> None:
    first = state.scenarios[0]
    second = state.scenarios[1]
    names = ("activations", "top-k IDs", "top-k weights")
    for name, first_tensor, second_tensor in zip(names, first, second, strict=True):
        if torch.equal(first_tensor, second_tensor):
            raise AssertionError(f"live-input scenarios have identical {name}")
    if torch.equal(state.references[0], state.references[1]):
        raise AssertionError("live-input scenarios have identical oracle outputs")


def _scenario_input_sha256(
    scenario: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> dict[str, str]:
    return {
        name: _tensor_sha256(tensor)
        for name, tensor in zip(
            ("activations", "topk_ids", "topk_weights"), scenario, strict=True
        )
    }


def _runtime_compile_args(
    *,
    E: int,
    m: int,
    k: int,
    n: int,
    num_topk: int,
    topk_ids_dtype: torch.dtype,
    fast_math: bool,
    activation: str,
    quant_mode: str,
    w4a8_repacked: bool,
    direct_routing: bool,
    share_input_across_experts: bool,
    deterministic_output: bool,
    swiglu_limit: float | None,
    swiglu_alpha: float | None,
    swiglu_beta: float | None,
) -> list[object]:
    quant_mode = tp_moe._normalize_quant_mode(quant_mode)
    is_w4a8 = tp_moe._is_w4a8_quant_mode(quant_mode)
    share_input_across_experts = bool(
        share_input_across_experts
        and (quant_mode == "nvfp4" or (quant_mode == "w4a8_mx" and w4a8_repacked))
    )
    activation_spec = tp_moe._get_activation_kernel_spec(
        activation, quant_mode=quant_mode
    )
    swiglu_limit, swiglu_alpha, swiglu_beta = tp_moe._normalize_swiglu_params(
        activation_spec.activation,
        swiglu_limit,
        swiglu_alpha,
        swiglu_beta,
    )
    tile = tp_moe._select_dynamic_tile_mn(
        m * num_topk,
        n,
        quant_mode,
        num_experts=E,
        activation=activation_spec.activation,
    )
    dynamic_down_scale = tp_moe._dynamic_down_scale_enabled() and not is_w4a8
    materialize_intermediate = tp_moe._w4a8_dynamic_materialized_enabled(
        quant_mode=quant_mode,
        activation=activation_spec.activation,
        num_tokens=m,
        routed_rows=m * num_topk,
        num_experts=E,
        k=k,
        n=n,
        w4a8_repacked=w4a8_repacked,
        share_input_across_experts=share_input_across_experts,
        deterministic_output=deterministic_output,
    )
    swap_ab = bool(
        quant_mode == "nvfp4"
        and activation_spec.is_gated
        and n % 128 != 0
        and n % 32 == 0
    )
    swap_env = os.environ.get("B12X_DYNAMIC_SWAP_AB")
    if swap_env is not None:
        swap_ab = swap_env != "0"
    if is_w4a8:
        swap_ab = False
    return [
        quant_mode,
        "dynamic",
        int(E),
        int(k),
        int(n),
        int(num_topk),
        list(tile),
        ["repr", "torch", "dtype", str(topk_ids_dtype)],
        bool(fast_math),
        activation_spec.activation,
        swiglu_limit,
        swiglu_alpha,
        swiglu_beta,
        bool(dynamic_down_scale),
        bool(share_input_across_experts),
        bool(swap_ab),
        bool(deterministic_output),
        tp_moe._dynamic_work_source(),
        bool(w4a8_repacked),
        bool(direct_routing),
        bool(materialize_intermediate),
    ]


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
    metadata: list[dict[str, object]] = []
    for index, node in enumerate(nodes):
        error, node_type = cuda_driver.cuGraphNodeGetType(node)
        if error != cuda_driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuGraphNodeGetType failed: {error}")
        item: dict[str, object] = {"index": index, "type": node_type.name}
        if node_type == cuda_driver.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL:
            error, params = cuda_driver.cuGraphKernelNodeGetParams(node)
            if error != cuda_driver.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"cuGraphKernelNodeGetParams failed: {error}")
            error, name = cuda_driver.cuFuncGetName(params.func)
            if error != cuda_driver.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"cuFuncGetName failed: {error}")
            item.update(
                {
                    "kernel_name": name.decode(),
                    "grid": [params.gridDimX, params.gridDimY, params.gridDimZ],
                    "block": [params.blockDimX, params.blockDimY, params.blockDimZ],
                    "dynamic_smem_bytes": params.sharedMemBytes,
                }
            )
        metadata.append(item)
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
            item["type"] == "CU_GRAPH_NODE_TYPE_KERNEL" for item in metadata
        ),
        "nodes": metadata,
        "edges": [
            [node_indices[int(source)], node_indices[int(destination)]]
            for source, destination in zip(sources, destinations, strict=True)
        ],
    }


def _topology_signature(topology: Mapping[str, Any]) -> dict[str, object]:
    return {
        "node_count": topology["node_count"],
        "kernel_node_count": topology["kernel_node_count"],
        "nodes": [
            {key: value for key, value in node.items() if key != "kernel_name"}
            for node in topology["nodes"]
        ],
        "edges": topology["edges"],
    }


def _validate_manifest_kernels(
    topology: Mapping[str, Any],
    launch_metadata: Mapping[str, Any],
) -> None:
    expected = launch_metadata["launch_dynamic_smem_bytes"]
    observed = {
        str(node["kernel_name"]): int(node["dynamic_smem_bytes"])
        for node in topology["nodes"]
        if node["type"] == "CU_GRAPH_NODE_TYPE_KERNEL"
    }
    missing = set(expected) - set(observed)
    if missing:
        raise AssertionError(f"graph is missing manifest kernels: {sorted(missing)}")
    for name, allowed_smem in expected.items():
        if observed[name] not in [int(value) for value in allowed_smem]:
            raise AssertionError(
                f"{name}: graph SMEM {observed[name]} is absent from manifest "
                f"values {allowed_smem}"
            )


def _correctness_metrics(
    actual: torch.Tensor,
    reference: torch.Tensor,
    case: DynamicCase,
) -> dict[str, object]:
    actual_fp32 = actual.float()
    reference_fp32 = reference.float()
    difference = actual_fp32 - reference_fp32
    reference_rms = float(reference_fp32.square().mean().sqrt().item())
    metrics = compare_to_reference(actual_fp32, reference_fp32)
    normalized_rmse = metrics.rmse / max(reference_rms, 1.0e-30)
    finite = bool(torch.isfinite(actual_fp32).all().item())
    nonzero = int(torch.count_nonzero(actual_fp32).item())
    passed = bool(
        finite
        and nonzero > 0
        and reference_rms > 1.0e-5
        and math.isfinite(metrics.cos)
        and metrics.cos >= case.min_cosine
        and math.isfinite(normalized_rmse)
        and normalized_rmse <= case.max_normalized_rmse
    )
    result = {
        "passed": passed,
        "finite": finite,
        "nonzero": nonzero,
        "max_abs": float(difference.abs().max().item()),
        "cosine": metrics.cos,
        "rmse": metrics.rmse,
        "reference_rms": reference_rms,
        "normalized_rmse": normalized_rmse,
        "sha256": _tensor_sha256(actual),
    }
    if not passed:
        raise AssertionError(f"{case.name}: oracle failure: {result}")
    return result


def _mode_snapshot(expected_physical_gpu: int) -> dict[str, object]:
    snapshot = nvidia_smi_gpu_mode_snapshot()
    if not snapshot.get("available"):
        raise RuntimeError(f"physical GPU mode snapshot unavailable: {snapshot}")
    fields_map = snapshot.get("fields")
    if not isinstance(fields_map, dict):
        raise RuntimeError(f"GPU mode snapshot has no fields: {snapshot}")
    observed = int(str(fields_map["index"]))
    if observed != expected_physical_gpu:
        raise RuntimeError(
            f"expected physical GPU {expected_physical_gpu}, observed {observed}"
        )
    return snapshot


def _run_case(
    *,
    case: DynamicCase,
    compile_args: list[object],
    compiled: Mapping[str, Any],
    provenance: Mapping[str, Mapping[str, Any]],
    labels: tuple[str, str],
    precondition_cycles: int,
    precondition_seconds: float,
    maximum_precondition_seconds: float,
    warmup_cycles: int,
    cycles: int,
    event_batch_cycles: int,
    replays_per_reported_sample: int,
    l2_flush_bytes: int,
    expected_physical_gpu: int,
    max_sm_clock_delta_mhz: float,
) -> dict[str, object]:
    with _specialization_environment(compile_args) as controlled_environment:
        state = _build_case_state(case, compile_args)
        torch.cuda.synchronize()
        _assert_scenarios_distinct(state)
        scenario_input_sha256 = tuple(
            _scenario_input_sha256(scenario) for scenario in state.scenarios
        )
        changed_inputs = {
            name: scenario_input_sha256[0][name] != scenario_input_sha256[1][name]
            for name in scenario_input_sha256[0]
        }
        if not all(changed_inputs.values()):
            raise AssertionError(
                f"live-input scenario hashes are not all distinct: {changed_inputs}"
            )
        # Populate the fixed flush-buffer cache before the outer allocator
        # baseline. The shared timer resolves and reuses this exact capacity.
        if make_l2_flush_fn(True, l2_flush_bytes) is None:
            raise AssertionError("cold-L2 flush was not constructed")

        stable_tensors = _stable_tensors(state)
        stable_pointers = {
            name: tensor.data_ptr() for name, tensor in stable_tensors.items()
        }
        immutable_tensors = _tensor_leaves(state.immutable_roots)
        immutable_before = {
            name: _tensor_sha256(tensor) for name, tensor in immutable_tensors.items()
        }
        artifact_before = {
            label: _verify_artifact(provenance[label]) for label in labels
        }

        active_label: str | None = None
        dispatch_records: dict[str, list[dict[str, object]]] = {
            label: [] for label in labels
        }
        original_get_dynamic_kernel = tp_moe._get_dynamic_kernel

        def select_arm(label: str | None) -> None:
            nonlocal active_label
            if label is not None and label not in labels:
                raise ValueError(f"unknown arm label {label!r}")
            active_label = label

        def exact_get_dynamic_kernel(
            E: int,
            m: int,
            k: int,
            n: int,
            num_topk: int,
            max_rows: int,
            *,
            topk_ids_dtype: torch.dtype,
            fast_math: bool,
            mac_override: int | None = None,
            activation: str = "silu",
            quant_mode: str = "nvfp4",
            w4a8_repacked: bool = False,
            direct_routing: bool = False,
            share_input_across_experts: bool = False,
            deterministic_output: bool = False,
            swiglu_limit: float | None = None,
            swiglu_alpha: float | None = None,
            swiglu_beta: float | None = None,
        ):
            if active_label is None:
                raise RuntimeError("exact TP-MoE dispatch has no selected arm")
            observed_args = _runtime_compile_args(
                E=E,
                m=m,
                k=k,
                n=n,
                num_topk=num_topk,
                topk_ids_dtype=topk_ids_dtype,
                fast_math=fast_math,
                activation=activation,
                quant_mode=quant_mode,
                w4a8_repacked=w4a8_repacked,
                direct_routing=direct_routing,
                share_input_across_experts=share_input_across_experts,
                deterministic_output=deterministic_output,
                swiglu_limit=swiglu_limit,
                swiglu_alpha=swiglu_alpha,
                swiglu_beta=swiglu_beta,
            )
            if observed_args != compile_args:
                mismatches = {
                    f"arg{index}": {
                        "manifest": compile_args[index],
                        "runtime": value,
                    }
                    for index, value in enumerate(observed_args)
                    if value != compile_args[index]
                }
                raise RuntimeError(
                    f"{case.name}: runtime dispatch specialization differs: {mismatches}"
                )
            mac = (
                int(mac_override)
                if mac_override is not None
                else int(tp_moe._get_impl_mac("dynamic"))
            )
            dispatch_records[active_label].append(
                {
                    "runtime_compile_args": observed_args,
                    "m": int(m),
                    "max_rows": int(max_rows),
                    "max_active_clusters": mac,
                    "object_sha256": provenance[active_label]["object_sha256"],
                }
            )
            return compiled[active_label], mac

        tp_moe._get_dynamic_kernel = exact_get_dynamic_kernel
        try:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                _install_scenario(state, 0)
            stream.synchronize()
            _assert_live_contract(state, 0, int(compile_args[2]))

            def launch_exact(label: str) -> torch.Tensor:
                select_arm(label)
                output = tp_moe.b12x_moe_fp4(binding=state.binding)
                if output.data_ptr() != state.output.data_ptr():
                    raise AssertionError(f"{label}: launcher replaced output")
                return output

            graphs: dict[str, torch.cuda.CUDAGraph] = {}
            topologies: dict[str, dict[str, object]] = {}
            for label in labels:
                with torch.cuda.stream(stream):
                    launch_exact(label)
                stream.synchronize()
                graph = torch.cuda.CUDAGraph(keep_graph=True)
                select_arm(label)
                with torch.cuda.graph(graph, stream=stream):
                    launch_exact(label)
                stream.synchronize()
                graphs[label] = graph
                topologies[label] = _graph_topology(graph)
                _validate_manifest_kernels(
                    topologies[label], provenance[label]["launch_metadata"]
                )
            select_arm(None)

            topology_equal = _topology_signature(
                topologies[labels[0]]
            ) == _topology_signature(topologies[labels[1]])
            if not topology_equal:
                raise AssertionError("arm CUDA graph topologies differ")
            for label in labels:
                if len(dispatch_records[label]) != 2:
                    raise AssertionError(
                        f"{label}: expected eager+capture exact dispatches, got "
                        f"{len(dispatch_records[label])}"
                    )

            correctness: dict[str, dict[str, object]] = {}
            output_by_stage: dict[str, dict[str, torch.Tensor]] = {}

            def assert_stable_pointers() -> None:
                observed = {
                    name: tensor.data_ptr() for name, tensor in stable_tensors.items()
                }
                if observed != stable_pointers:
                    raise AssertionError(
                        f"stable tensor pointers changed: {stable_pointers} -> {observed}"
                    )

            def bf16_ordered(tensor: torch.Tensor) -> torch.Tensor:
                if tensor.dtype is not torch.bfloat16:
                    raise TypeError(
                        f"BF16 ULP accounting requires bfloat16, got {tensor.dtype}"
                    )
                bits = tensor.contiguous().view(torch.int16).to(torch.int32)
                unsigned = bits & 0xFFFF
                magnitude = unsigned & 0x7FFF
                return torch.where(
                    (unsigned & 0x8000) != 0,
                    0x8000 - magnitude,
                    0x8000 + magnitude,
                )

            def replay_envelope(
                samples: list[torch.Tensor],
            ) -> tuple[
                dict[str, object],
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
            ]:
                if len(samples) < 2:
                    raise AssertionError("a replay envelope requires multiple samples")
                value_min = samples[0].clone()
                value_max = samples[0].clone()
                ordered_min = bf16_ordered(samples[0])
                ordered_max = ordered_min.clone()
                for sample in samples[1:]:
                    value_min = torch.minimum(value_min, sample)
                    value_max = torch.maximum(value_max, sample)
                    ordered = bf16_ordered(sample)
                    ordered_min = torch.minimum(ordered_min, ordered)
                    ordered_max = torch.maximum(ordered_max, ordered)
                ulp_range = ordered_max - ordered_min
                value_range = value_max.float() - value_min.float()
                metadata = {
                    "replay_count": len(samples),
                    "max_abs_range": float(value_range.max().item()),
                    "max_bf16_ulp_range": int(ulp_range.max().item()),
                    "varying_elements": int((ulp_range != 0).sum().item()),
                }
                return (
                    metadata,
                    ordered_min,
                    ordered_max,
                    value_min,
                    value_max,
                    ulp_range,
                )

            def validate(stage: str, scenario: int) -> None:
                with torch.cuda.stream(stream):
                    _install_scenario(state, scenario)
                stream.synchronize()
                _assert_live_contract(state, scenario, int(compile_args[2]))
                assert_stable_pointers()
                stage_outputs: dict[str, torch.Tensor] = {}
                stage_result: dict[str, object] = {
                    "scenario": scenario,
                    "arms": {},
                }
                reference_quantized = state.references[scenario].to(
                    dtype=state.output.dtype
                )
                nondeterministic_exception = not bool(compile_args[16])
                sentinel_specs = (
                    ("nan", math.nan),
                    ("finite_high_positive", 997.0),
                    ("finite_high_negative", -733.0),
                )
                sentinel_rounds = 2 if nondeterministic_exception else 1
                replay_outputs: dict[str, list[torch.Tensor]] = {}
                for label in labels:
                    sentinel_replays: list[dict[str, object]] = []
                    arm_replay_outputs: list[torch.Tensor] = []
                    canonical_output: torch.Tensor | None = None
                    canonical_metrics: dict[str, object] | None = None
                    sentinel_outputs_bit_exact = True
                    sentinel_max_abs_from_canonical = 0.0
                    sentinel_schedule = sentinel_specs * sentinel_rounds
                    for replay_index, (sentinel_name, sentinel) in enumerate(
                        sentinel_schedule
                    ):
                        with torch.cuda.stream(stream):
                            state.output.fill_(sentinel)
                        stream.synchronize()
                        allocated_before = torch.cuda.memory_allocated()
                        reserved_before = torch.cuda.memory_reserved()
                        with torch.cuda.stream(stream):
                            graphs[label].replay()
                        stream.synchronize()
                        allocated_after = torch.cuda.memory_allocated()
                        reserved_after = torch.cuda.memory_reserved()
                        if (allocated_before, reserved_before) != (
                            allocated_after,
                            reserved_after,
                        ):
                            raise AssertionError(
                                f"{case.name}/{label}/{sentinel_name}: replay "
                                "allocated memory: "
                                f"{(allocated_before, reserved_before)} -> "
                                f"{(allocated_after, reserved_after)}"
                            )
                        assert_stable_pointers()

                        if math.isnan(sentinel):
                            observed_mask = torch.isnan(state.output)
                            reference_mask = torch.isnan(reference_quantized)
                            sentinel_value: str | float = "nan"
                        else:
                            observed_mask = state.output == sentinel
                            reference_mask = reference_quantized == sentinel
                            sentinel_value = float(
                                state.output.new_tensor(sentinel).item()
                            )
                        observed_count = int(observed_mask.sum().item())
                        reference_count = int(reference_mask.sum().item())
                        overlap_count = int(
                            (observed_mask & reference_mask).sum().item()
                        )
                        unexplained_count = int(
                            (observed_mask & ~reference_mask).sum().item()
                        )
                        sample_coordinates = (
                            torch.nonzero(observed_mask, as_tuple=False)[:8]
                            .cpu()
                            .tolist()
                        )
                        if reference_count:
                            raise AssertionError(
                                f"{case.name}: selected {sentinel_name} sentinel "
                                f"collides with {reference_count} quantized oracle "
                                "elements"
                            )
                        if observed_count:
                            raise AssertionError(
                                f"{case.name}/{label}: {observed_count} "
                                f"{sentinel_name} sentinel elements remained after "
                                "graph replay"
                            )

                        replay_output = state.output.clone()
                        arm_replay_outputs.append(replay_output)
                        replay_metrics = _correctness_metrics(
                            replay_output, state.references[scenario], case
                        )
                        if canonical_output is None:
                            canonical_output = replay_output
                            canonical_metrics = replay_metrics
                            max_abs_from_canonical = 0.0
                        else:
                            delta = replay_output.float() - canonical_output.float()
                            max_abs_from_canonical = float(delta.abs().max().item())
                            sentinel_max_abs_from_canonical = max(
                                sentinel_max_abs_from_canonical,
                                max_abs_from_canonical,
                            )
                            sentinel_outputs_bit_exact &= torch.equal(
                                canonical_output, replay_output
                            )
                        sentinel_replays.append(
                            {
                                "replay_index": replay_index,
                                "sentinel_round": replay_index // len(sentinel_specs),
                                "name": sentinel_name,
                                "value": sentinel_value,
                                "observed_sentinel_count": observed_count,
                                "quantized_reference_sentinel_count": (reference_count),
                                "observed_reference_overlap_count": overlap_count,
                                "observed_not_in_reference_count": unexplained_count,
                                "observed_sample_coordinates": sample_coordinates,
                                "correctness": replay_metrics,
                                "max_abs_from_nan_sentinel_output": (
                                    max_abs_from_canonical
                                ),
                                "allocator_before": {
                                    "allocated": allocated_before,
                                    "reserved": reserved_before,
                                },
                                "allocator_after": {
                                    "allocated": allocated_after,
                                    "reserved": reserved_after,
                                },
                            }
                        )

                    if canonical_output is None or canonical_metrics is None:
                        raise AssertionError("sentinel replay set is empty")
                    if (
                        not nondeterministic_exception
                        and not sentinel_outputs_bit_exact
                    ):
                        raise AssertionError(
                            f"{case.name}/{label}: strict replay outputs are not bit "
                            f"exact; max_abs={sentinel_max_abs_from_canonical}"
                        )
                    stage_outputs[label] = canonical_output
                    replay_outputs[label] = arm_replay_outputs
                    stage_result["arms"][label] = {
                        **canonical_metrics,
                        "full_output_overwrite_proven": True,
                        "sentinel_outputs_bit_exact": sentinel_outputs_bit_exact,
                        "sentinel_max_abs_from_nan_output": (
                            sentinel_max_abs_from_canonical
                        ),
                        "sentinel_replays": sentinel_replays,
                    }
                arms_bit_exact = torch.equal(
                    stage_outputs[labels[0]], stage_outputs[labels[1]]
                )
                canonical_delta = (
                    stage_outputs[labels[1]].float() - stage_outputs[labels[0]].float()
                )
                canonical_max_abs = float(canonical_delta.abs().max().item())
                if nondeterministic_exception:
                    arm_envelopes: dict[str, dict[str, object]] = {}
                    private_envelopes: dict[
                        str,
                        tuple[
                            torch.Tensor,
                            torch.Tensor,
                            torch.Tensor,
                            torch.Tensor,
                            torch.Tensor,
                        ],
                    ] = {}
                    for label in labels:
                        (
                            envelope_metadata,
                            ordered_min,
                            ordered_max,
                            value_min,
                            value_max,
                            ulp_range,
                        ) = replay_envelope(replay_outputs[label])
                        arm_envelopes[label] = envelope_metadata
                        private_envelopes[label] = (
                            ordered_min,
                            ordered_max,
                            value_min,
                            value_max,
                            ulp_range,
                        )

                    a_private = private_envelopes[labels[0]]
                    b_private = private_envelopes[labels[1]]
                    union_ordered_min = torch.minimum(a_private[0], b_private[0])
                    union_ordered_max = torch.maximum(a_private[1], b_private[1])
                    cross_arm_ulp_range = union_ordered_max - union_ordered_min
                    same_arm_union_ulp_range = torch.maximum(a_private[4], b_private[4])
                    exact_bf16_ulp_margin = 1
                    allowed_ulp_range = same_arm_union_ulp_range + exact_bf16_ulp_margin
                    violations = cross_arm_ulp_range > allowed_ulp_range
                    violation_count = int(violations.sum().item())
                    violation_samples = (
                        torch.nonzero(violations, as_tuple=False)[:8].cpu().tolist()
                    )
                    union_value_min = torch.minimum(a_private[2], b_private[2])
                    union_value_max = torch.maximum(a_private[3], b_private[3])
                    cross_arm_abs_range = (
                        union_value_max.float() - union_value_min.float()
                    )
                    flat_cross_max_index = int(cross_arm_abs_range.argmax().item())
                    cross_arm_max_abs_range = float(
                        cross_arm_abs_range.flatten()[flat_cross_max_index].item()
                    )
                    coordinate: list[int] = []
                    remaining_index = flat_cross_max_index
                    for extent in reversed(cross_arm_abs_range.shape):
                        remaining_index, component = divmod(
                            remaining_index, int(extent)
                        )
                        coordinate.append(component)
                    coordinate.reverse()
                    cross_endpoints = torch.stack(
                        (
                            union_value_min.flatten()[flat_cross_max_index],
                            union_value_max.flatten()[flat_cross_max_index],
                        )
                    )
                    toward_positive = torch.nextafter(
                        cross_endpoints,
                        torch.full_like(cross_endpoints, math.inf),
                    )
                    toward_negative = torch.nextafter(
                        cross_endpoints,
                        torch.full_like(cross_endpoints, -math.inf),
                    )
                    exact_bf16_ulp_at_cross_max = float(
                        torch.maximum(
                            (toward_positive.float() - cross_endpoints.float()).abs(),
                            (cross_endpoints.float() - toward_negative.float()).abs(),
                        )
                        .max()
                        .item()
                    )
                    same_arm_union_max_abs_range = max(
                        float(arm_envelopes[label]["max_abs_range"]) for label in labels
                    )
                    scalar_abs_limit = (
                        same_arm_union_max_abs_range + exact_bf16_ulp_at_cross_max
                    )
                    scalar_envelope_passed = cross_arm_max_abs_range <= scalar_abs_limit
                    elementwise_envelope_passed = violation_count == 0
                    arm_policy = {
                        "kind": (
                            "manifest-declared-nondeterministic-bf16-max-abs-envelope"
                        ),
                        "manifest_deterministic_output": bool(compile_args[16]),
                        "replays_per_arm": len(replay_outputs[labels[0]]),
                        "sentinel_rounds": sentinel_rounds,
                        "same_arm_envelopes": arm_envelopes,
                        "canonical_arms_bit_exact": arms_bit_exact,
                        "canonical_cross_arm_max_abs": canonical_max_abs,
                        "cross_arm_union_max_abs_range": cross_arm_max_abs_range,
                        "cross_arm_max_abs_coordinate": coordinate,
                        "cross_arm_max_abs_endpoints": [
                            float(value.float().item()) for value in cross_endpoints
                        ],
                        "same_arm_union_max_abs_range": (same_arm_union_max_abs_range),
                        "exact_bf16_ulp_at_cross_max": (exact_bf16_ulp_at_cross_max),
                        "scalar_abs_envelope_limit": scalar_abs_limit,
                        "scalar_abs_envelope_passed": scalar_envelope_passed,
                        "cross_arm_union_max_bf16_ulp_range": int(
                            cross_arm_ulp_range.max().item()
                        ),
                        "same_arm_union_max_bf16_ulp_range": int(
                            same_arm_union_ulp_range.max().item()
                        ),
                        "exact_bf16_ulp_margin": exact_bf16_ulp_margin,
                        "elementwise_violation_count": violation_count,
                        "violation_sample_coordinates": violation_samples,
                        "elementwise_ulp_diagnostic_passed": (
                            elementwise_envelope_passed
                        ),
                        "passed": (
                            scalar_envelope_passed and elementwise_envelope_passed
                        ),
                    }
                    if not arm_policy["passed"]:
                        raise AssertionError(
                            f"{case.name}/{stage}: cross-arm replay envelope failed; "
                            f"max_abs={cross_arm_max_abs_range}, intrinsic_abs="
                            f"{same_arm_union_max_abs_range}, exact_bf16_ulp="
                            f"{exact_bf16_ulp_at_cross_max}, coordinate={coordinate}, "
                            f"elementwise_ulp_violations={violation_count}"
                        )
                else:
                    if not arms_bit_exact:
                        raise AssertionError(
                            f"{case.name}/{stage}: arms are not bit exact; "
                            f"max_abs={canonical_max_abs}"
                        )
                    arm_policy = {
                        "kind": "strict-bit-exact",
                        "manifest_deterministic_output": bool(compile_args[16]),
                        "empirically_bit_exact_required": True,
                        "canonical_cross_arm_max_abs": canonical_max_abs,
                        "passed": True,
                    }
                stage_result.update(
                    {
                        "arms_bit_exact": arms_bit_exact,
                        "arm_comparison_policy": arm_policy,
                        "passed": True,
                    }
                )
                correctness[stage] = stage_result
                output_by_stage[stage] = stage_outputs

            validate("pre_timing", 0)
            timing_live_inputs_before = {
                "activations": _tensor_sha256(state.live_a),
                "topk_ids": _tensor_sha256(state.live_topk_ids),
                "topk_weights": _tensor_sha256(state.live_topk_weights),
            }

            allocation_before_timing = allocator_counters()
            conditions = time_conditions(
                graphs,
                labels=labels,
                precondition=precondition_cycles,
                warmup=warmup_cycles,
                cycles=cycles,
                event_batch_cycles=event_batch_cycles,
                stream=stream,
                cold_l2=True,
                l2_flush_bytes=l2_flush_bytes,
                replays_per_reported_sample=replays_per_reported_sample,
                precondition_seconds=precondition_seconds,
                maximum_precondition_seconds=maximum_precondition_seconds,
                mode_snapshot=lambda: _mode_snapshot(expected_physical_gpu),
                required_pstate="P1",
                max_sm_clock_delta_mhz=max_sm_clock_delta_mhz,
            )
            allocation_after_timing = allocator_counters()
            if allocation_after_timing != allocation_before_timing:
                raise AssertionError(
                    "CUDA allocator state changed across graph timing: "
                    f"before={allocation_before_timing}, "
                    f"after={allocation_after_timing}"
                )
            compile_spec_hashes = {
                label: str(provenance[label]["compile_spec_hash"]) for label in labels
            }
            all_graph_spec_hashes = sorted(set(compile_spec_hashes.values()))
            if all_graph_spec_hashes != [case.spec_hash]:
                raise AssertionError(
                    f"{case.name}: timed graph compile-spec coverage changed: "
                    f"{all_graph_spec_hashes}"
                )
            for condition in conditions.values():
                condition["compile_spec_hashes"] = dict(compile_spec_hashes)
                condition["all_graph_spec_hashes"] = list(all_graph_spec_hashes)

            timing_live_inputs_after = {
                "activations": _tensor_sha256(state.live_a),
                "topk_ids": _tensor_sha256(state.live_topk_ids),
                "topk_weights": _tensor_sha256(state.live_topk_weights),
            }
            if timing_live_inputs_after != timing_live_inputs_before:
                raise AssertionError("timed graph replay mutated live inputs")

            validate("post_timing_live_mutation", 1)
            scenario_0_outputs = output_by_stage["pre_timing"]
            scenario_1_outputs = output_by_stage["post_timing_live_mutation"]
            changed_outputs = {
                label: not torch.equal(
                    scenario_0_outputs[label], scenario_1_outputs[label]
                )
                for label in labels
            }
            unchanged_output_arms = [
                label for label, changed in changed_outputs.items() if not changed
            ]
            if unchanged_output_arms:
                raise AssertionError(
                    "live-input mutation did not change every arm output: "
                    f"{unchanged_output_arms}"
                )
            scenario_0_output_sha256 = {
                label: _tensor_sha256(scenario_0_outputs[label]) for label in labels
            }
            scenario_1_output_sha256 = {
                label: _tensor_sha256(scenario_1_outputs[label]) for label in labels
            }
            if any(
                scenario_0_output_sha256[label] == scenario_1_output_sha256[label]
                for label in labels
            ):
                raise AssertionError(
                    "live-input mutation produced an identical per-arm output hash"
                )

            immutable_after = {
                name: _tensor_sha256(tensor)
                for name, tensor in immutable_tensors.items()
            }
            if immutable_after != immutable_before:
                raise AssertionError("read-only tensors changed during benchmark")
            assert_stable_pointers()
            artifact_after = {
                label: _verify_artifact(provenance[label]) for label in labels
            }
            allocator_checks = {
                stage: {
                    label: [
                        {
                            "replay_index": replay["replay_index"],
                            "sentinel": replay["name"],
                            "before": replay["allocator_before"],
                            "after": replay["allocator_after"],
                        }
                        for replay in stage_data["arms"][label]["sentinel_replays"]
                    ]
                    for label in labels
                }
                for stage, stage_data in correctness.items()
            }
            return {
                "name": case.name,
                "compile_spec_hash": case.spec_hash,
                "compile_args": compile_args,
                "corpus_nodeid": case.corpus_nodeid,
                "shape": {
                    "m": case.m,
                    "experts": compile_args[2],
                    "k": compile_args[3],
                    "n": compile_args[4],
                    "topk": compile_args[5],
                    "tile": compile_args[6],
                    "quant_mode": compile_args[0],
                    "activation": compile_args[9],
                },
                "controlled_specialization_environment": controlled_environment,
                "object_provenance": provenance,
                "artifact_verification_before": artifact_before,
                "artifact_verification_after": artifact_after,
                "dispatch_records": dispatch_records,
                "same_address_arms": True,
                "fixed_workspace": {
                    "verified": True,
                    "scratch_caps_frozen": state.scratch_plan.caps.frozen,
                    "scratch_specs": [
                        {
                            "shape": list(spec.shape),
                            "dtype": str(spec.dtype),
                            "device": str(spec.device),
                        }
                        for spec in state.scratch_plan.scratch_specs()
                    ],
                    "stable_tensor_count": len(stable_tensors),
                    "stable_pointers": stable_pointers,
                },
                "read_only_inputs_unchanged": True,
                "read_only_inputs_immutable": True,
                "read_only_tensor_sha256": immutable_before,
                "read_only_inputs": {
                    "unchanged": True,
                    "sha256_before": immutable_before,
                    "sha256_after": immutable_after,
                    "timed_live_scenario_0": {
                        "unchanged": True,
                        "sha256_before": timing_live_inputs_before,
                        "sha256_after": timing_live_inputs_after,
                    },
                },
                "poisoned_outputs_overwritten": True,
                "cuda_graph_replay": True,
                "cuda_graph_topology": topologies,
                "cuda_graph_topology_equal": topology_equal,
                "live_input_scenarios_distinct": True,
                "live_input_mutation_changed_input": True,
                "live_input_mutation_changed_output": True,
                "live_input_mutation": {
                    "captured_graph_reused": True,
                    "in_place": True,
                    "same_addresses": True,
                    "allocation_addresses_stable": True,
                    "scenarios_distinct": True,
                    "changed_input": True,
                    "changed_inputs": changed_inputs,
                    "changed_output": True,
                    "changed_outputs": changed_outputs,
                    "scenario_0_sha256": scenario_input_sha256[0],
                    "scenario_1_sha256": scenario_input_sha256[1],
                    "scenario_0_output_sha256": scenario_0_output_sha256,
                    "scenario_1_output_sha256": scenario_1_output_sha256,
                    "activations": True,
                    "topk_ids": True,
                    "topk_weights": True,
                    "timing_inputs_unchanged": True,
                    "scenario_0_sha256_before_timing": timing_live_inputs_before,
                    "scenario_0_sha256_after_timing": timing_live_inputs_after,
                },
                "allocator_stable": True,
                "zero_replay_allocations": True,
                "allocator_checks": allocator_checks,
                "timing_allocator_before": allocation_before_timing,
                "timing_allocator_after": allocation_after_timing,
                "correctness": correctness,
                "precondition_cycles_minimum": precondition_cycles,
                "precondition_seconds_minimum": precondition_seconds,
                "maximum_precondition_seconds": maximum_precondition_seconds,
                "warmup_cycles_per_condition": warmup_cycles,
                "timed_abba_cycles_per_condition": cycles,
                "event_batch_cycles": event_batch_cycles,
                "replays_per_reported_sample": replays_per_reported_sample,
                "timed_reported_samples_per_arm_per_condition": cycles * 2,
                "timed_inner_replays_per_arm_per_condition": (
                    cycles * 2 * replays_per_reported_sample
                ),
                "required_pstate": "P1",
                "required_active_throttle_reasons": 0,
                "max_sm_clock_delta_mhz": max_sm_clock_delta_mhz,
                "required_l2_conditions": ["warm_l2", "cold_l2"],
                "conditions": conditions,
            }
        finally:
            tp_moe._get_dynamic_kernel = original_get_dynamic_kernel


def main() -> None:
    args = _args()
    if args.cycles < 500 or args.cycles % 2:
        raise ValueError("--cycles must be an even integer of at least 500")
    if args.precondition_cycles < 1:
        raise ValueError("--precondition-cycles must be positive")
    if args.warmup_cycles < 1 or args.event_batch_cycles < 1:
        raise ValueError("warmup and event-batch cycles must be positive")
    if args.replays_per_reported_sample < 1:
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
    current_package_fingerprint = cute_compiler.b12x_package_fingerprint()
    if current_package_fingerprint != args.expected_current_package_fingerprint:
        raise RuntimeError(
            "current b12x package fingerprint differs from the explicit host-source "
            "pin: "
            f"expected={args.expected_current_package_fingerprint}, "
            f"observed={current_package_fingerprint}"
        )
    current_relevant_source_sha256 = {
        "integration_tp_moe": _sha256_file(REPO_ROOT / "b12x/integration/tp_moe.py"),
        "dynamic_kernel": _sha256_file(REPO_ROOT / "b12x/moe/fused/dynamic.py"),
    }
    if current_relevant_source_sha256 != _FROZEN_RELEVANT_SOURCE_SHA256:
        raise RuntimeError(
            "current TP-MoE host sources differ from the frozen-v5 object source "
            "snapshot: "
            f"expected={_FROZEN_RELEVANT_SOURCE_SHA256}, "
            f"observed={current_relevant_source_sha256}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError(
            f"SM120 is required, got {torch.cuda.get_device_capability()}"
        )
    labels = (args.a_label, args.b_label)
    if labels[0] == labels[1]:
        raise ValueError("arm labels must differ")
    selected = tuple(dict.fromkeys(args.case or tuple(_CASES)))
    a_keys = _key_by_case(args.a_key, selected)
    b_keys = _key_by_case(args.b_key, selected)

    initial_mode = _mode_snapshot(args.expected_physical_gpu)
    cases: list[dict[str, object]] = []
    frozen_package_fingerprints: set[str] = set()
    for case_name in selected:
        case = _CASES[case_name]
        compiled_a, provenance_a = _load_exact(
            args.a_cache,
            case,
            a_keys.get(case_name),
            args.a_cutlass_version,
        )
        compiled_b, provenance_b = _load_exact(
            args.b_cache,
            case,
            b_keys.get(case_name),
            args.b_cutlass_version,
        )
        compiled = {labels[0]: compiled_a, labels[1]: compiled_b}
        provenance = {labels[0]: provenance_a, labels[1]: provenance_b}
        compile_args = _validate_pair(case, provenance, labels)
        frozen_package_fingerprints.update(
            str(arm["package_fingerprint"]) for arm in provenance.values()
        )
        cases.append(
            _run_case(
                case=case,
                compile_args=compile_args,
                compiled=compiled,
                provenance=provenance,
                labels=labels,
                precondition_cycles=args.precondition_cycles,
                precondition_seconds=args.precondition_seconds,
                maximum_precondition_seconds=args.maximum_precondition_seconds,
                warmup_cycles=args.warmup_cycles,
                cycles=args.cycles,
                event_batch_cycles=args.event_batch_cycles,
                replays_per_reported_sample=args.replays_per_reported_sample,
                l2_flush_bytes=args.l2_flush_bytes,
                expected_physical_gpu=args.expected_physical_gpu,
                max_sm_clock_delta_mhz=args.max_sm_clock_delta_mhz,
            )
        )
    if len(frozen_package_fingerprints) != 1:
        raise RuntimeError(
            "selected frozen artifacts do not share one package fingerprint: "
            f"{sorted(frozen_package_fingerprints)}"
        )
    final_mode = _mode_snapshot(args.expected_physical_gpu)

    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    all_correct = all(
        all(stage["passed"] for stage in case["correctness"].values()) for case in cases
    )
    all_arm_outputs_bit_exact = all(
        all(stage["arms_bit_exact"] for stage in case["correctness"].values())
        for case in cases
    )
    all_arm_output_policies_passed = all(
        all(
            stage["arm_comparison_policy"]["passed"]
            for stage in case["correctness"].values()
        )
        for case in cases
    )
    all_graph_topologies_equal = all(
        case["cuda_graph_topology_equal"] for case in cases
    )
    all_timing_conditions_passed = all(
        set(case["conditions"]) == {"warm_l2", "cold_l2"}
        and all(
            condition["allocator_stable"] is True
            and condition["gpu_mode_stability"]["passed"] is True
            for condition in case["conditions"].values()
        )
        for case in cases
    )
    result = {
        "schema": "b12x.tp_moe.dynamic.cache_abba.v2",
        "evidence_status": args.evidence_status,
        "evidence_scope": {
            "kind": "frozen-object/current-host-replay",
            "final_current_source_object_comparison": False,
            "note": (
                "Interim exact frozen-object evidence; final migration proof requires "
                "fresh 4.5.2 and 4.6.0 objects from the final source snapshot."
            ),
        },
        "command": [sys.executable, *sys.argv],
        "worktree": str(REPO_ROOT),
        "git_head": _git_output("rev-parse", "HEAD"),
        "git_status_porcelain": _git_output("status", "--short"),
        "source_sha256": {
            **current_relevant_source_sha256,
            "benchmark": _sha256_file(Path(__file__).resolve()),
        },
        "host_source_provenance": {
            "expected_current_package_fingerprint": (
                args.expected_current_package_fingerprint
            ),
            "observed_current_package_fingerprint": current_package_fingerprint,
            "frozen_artifact_package_fingerprint": next(
                iter(frozen_package_fingerprints)
            ),
            "current_matches_frozen_artifacts": (
                current_package_fingerprint in frozen_package_fingerprints
            ),
            "frozen_source_manifest_sha256": (_FROZEN_SOURCE_MANIFEST_SHA256),
            "frozen_relevant_source_sha256": (_FROZEN_RELEVANT_SOURCE_SHA256),
            "current_relevant_source_sha256": (current_relevant_source_sha256),
            "current_relevant_sources_match_frozen": True,
        },
        "gpu": {
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "expected_physical_index": args.expected_physical_gpu,
            "name": properties.name,
            "uuid": str(getattr(properties, "uuid", "")),
            "sms": properties.multi_processor_count,
            "capability": list(torch.cuda.get_device_capability()),
        },
        "gpu_mode_initial": initial_mode,
        "gpu_mode_final": final_mode,
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "nvidia_cutlass_dsl": _package_version("nvidia-cutlass-dsl"),
            "nvidia_cutlass_dsl_libs_base": _package_version(
                "nvidia-cutlass-dsl-libs-base"
            ),
            "nvidia_cutlass_dsl_libs_core": _package_version(
                "nvidia-cutlass-dsl-libs-core"
            ),
            "nvidia_cutlass_dsl_libs_cu12": _package_version(
                "nvidia-cutlass-dsl-libs-cu12"
            ),
            "nvidia_cutlass_dsl_libs_cu13": _package_version(
                "nvidia-cutlass-dsl-libs-cu13"
            ),
        },
        "arms": {
            labels[0]: {"expected_cutlass_dsl": args.a_cutlass_version},
            labels[1]: {"expected_cutlass_dsl": args.b_cutlass_version},
        },
        "cases": cases,
        "all_correct": all_correct,
        "all_arm_outputs_bit_exact": all_arm_outputs_bit_exact,
        "all_arm_output_policies_passed": all_arm_output_policies_passed,
        "all_graph_topologies_equal": all_graph_topologies_equal,
        "all_timing_conditions_passed": all_timing_conditions_passed,
        "all_evidence_gates_passed": (
            all_correct
            and all_arm_output_policies_passed
            and all_graph_topologies_equal
            and all_timing_conditions_passed
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "gpu": result["gpu"],
                "all_correct": result["all_correct"],
                "all_arm_outputs_bit_exact": result["all_arm_outputs_bit_exact"],
                "all_arm_output_policies_passed": result[
                    "all_arm_output_policies_passed"
                ],
                "all_graph_topologies_equal": result["all_graph_topologies_equal"],
                "all_timing_conditions_passed": result["all_timing_conditions_passed"],
                "ratios_b_over_a": {
                    case["name"]: {
                        condition: data["timings"]["ratios_b_over_a"]
                        for condition, data in case["conditions"].items()
                    }
                    for case in cases
                },
            },
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
