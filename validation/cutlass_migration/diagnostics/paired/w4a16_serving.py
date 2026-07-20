#!/usr/bin/env python3
"""Fail-closed exact-object CUDA-graph ABBA for W4A16 serving."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterator, Mapping

import torch

from benchmarks.common import nvidia_smi_gpu_mode_snapshot, resolve_l2_flush_bytes
from validation.cutlass_migration.core.evidence_status import (
    add_evidence_status_argument,
)
from validation.cutlass_migration.core.gpu_scope import (
    add_target_gpu_argument,
    require_target_gpu,
)
from validation.cutlass_migration.core.exact_cache_abba import (
    allocator_counters,
    graph_topology,
    load_exact,
    sha256_file,
    tensor_sha256,
    time_conditions,
    timing_mode_policy,
    topology_signature,
    verify_artifact,
)
from validation.cutlass_migration.paths import CORE_ROOT, REPO_ROOT
import b12x.cute.compiler as cute_compiler
from b12x.cute.intrinsics import swizzle_block_scale
from b12x.cute.runtime_control import (
    freeze_kernel_resolution,
    kernel_resolution_frozen,
    unfreeze_kernel_resolution,
)
from b12x.moe.fused.w4a16.host import (
    plan_w4a16_buffers,
    select_route_block_size_m,
)
import b12x.moe.fused.w4a16.kernel as w4a16_kernel
from b12x.moe.fused.w4a16.prepare import (
    make_w4a16_packed_buffers,
    prepare_w4a16_modelopt_nvfp4_weights,
)
from tests.w4a16_reference import moe_reference_w4a16


_EXPERTS = 128
_HIDDEN = 2688
_INTERMEDIATE = 1856
_TOPK = 6
_ACTIVATION = "relu2"
_REQUIRED_DECODE_M = (1, 2, 4, 8, 23, 33, 80)
_REQUIRED_PREFILL_M = (8192, 16384, 24576, 32768)
_REQUIRED_M = (*_REQUIRED_DECODE_M, *_REQUIRED_PREFILL_M)
_FUSED_SPEC_HASHES = {
    "direct": "a815e88a72ceadacf397843229aae2b9e195def107ba3cc25b50bca88d25100d",
    "routed": "768ad70185a1a96c0e1774e2c4f99e8848c95c4401a21e9a1e6d3d99bc2b2254",
    "prefill": "367125735a16899e06b2fd95e69591044c9cb51405511862e0efd6f59ffa6cc0",
}
_TOPK_SPEC_HASH = "26becba919aca623e3334925af43ccaa22df81301f202f840561c942fa700a34"
_CUTLASS_COMPONENTS = {
    "cutlass_dsl": "nvidia-cutlass-dsl",
    "cutlass_dsl_libs_base": "nvidia-cutlass-dsl-libs-base",
    "cutlass_dsl_libs_core": "nvidia-cutlass-dsl-libs-core",
    "cutlass_dsl_libs_cu12": "nvidia-cutlass-dsl-libs-cu12",
    "cutlass_dsl_libs_cu13": "nvidia-cutlass-dsl-libs-cu13",
}


def _active_throttle_reasons_argument(value: str) -> int:
    try:
        parsed = int(value, 0)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "active throttle-reasons mask must be 0 or 0x4"
        ) from error
    if parsed not in (0, 0x4):
        raise argparse.ArgumentTypeError(
            "active throttle-reasons mask must be 0 or 0x4"
        )
    return parsed


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_evidence_status_argument(parser)
    add_target_gpu_argument(parser)
    parser.add_argument("--a-cache", type=Path, required=True)
    parser.add_argument("--a-fingerprint", required=True)
    parser.add_argument("--a-label", default="cutlass-4.5.2")
    parser.add_argument("--a-cutlass-version", default="4.5.2")
    parser.add_argument("--a-topk-cache-key")
    parser.add_argument("--a-force-routed", action="store_true")
    parser.add_argument("--b-cache", type=Path, required=True)
    parser.add_argument("--b-fingerprint", required=True)
    parser.add_argument("--b-label", default="cutlass-4.6.0")
    parser.add_argument("--b-cutlass-version", default="4.6.0")
    parser.add_argument("--b-topk-cache-key")
    parser.add_argument("--b-force-routed", action="store_true")
    parser.add_argument(
        "--shared-topk-cache",
        type=Path,
        help="use one exact top-k object for both arms in a fused-only experiment",
    )
    parser.add_argument("--shared-topk-cache-key")
    parser.add_argument("--shared-topk-fingerprint")
    parser.add_argument("--shared-topk-cutlass-version")
    parser.add_argument(
        "--m",
        type=int,
        action="append",
        help="repeat for the exact required matrix; omission selects the chosen scope",
    )
    parser.add_argument(
        "--matrix-scope",
        choices=("full", "decode", "prefill"),
        default="full",
        help="explicitly run the full corpus or one complete serving regime",
    )
    parser.add_argument(
        "--precondition",
        type=int,
        default=1,
        help=(
            "minimum balanced ABBA cycles before the duration/mode gate; long "
            "prefill uses the five-second duration floor rather than a fixed "
            "1,000-cycle warmup"
        ),
    )
    parser.add_argument("--precondition-seconds", type=float, default=5.0)
    parser.add_argument("--maximum-precondition-seconds", type=float, default=30.0)
    parser.add_argument("--max-sm-clock-delta-mhz", type=float, default=60.0)
    parser.add_argument(
        "--required-active-throttle-reasons",
        type=_active_throttle_reasons_argument,
        default=0,
        metavar="{0,0x4}",
        help=(
            "exact active throttle-reasons mask required at precondition probe "
            "and before/after timing; 0x4 permits stable NVIDIA SW power cap"
        ),
    )
    parser.add_argument(
        "--allow-sw-power-cap-transition",
        action="store_true",
        help=(
            "diagnostic-only: permit observed 0x0/0x4 transitions while "
            "still requiring P1 and stable SM/memory clocks"
        ),
    )
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--cycles", type=int, default=500)
    parser.add_argument("--prefill-cycles", type=int, default=500)
    parser.add_argument("--event-batch-cycles", type=int, default=50)
    parser.add_argument("--l2-flush-bytes", type=int, default=0)
    parser.add_argument(
        "--cold-l2",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="required; collect explicit warm-L2 and cold-L2 conditions",
    )
    parser.add_argument(
        "--graph-dump-dir",
        type=Path,
        help="optionally dump each captured CUDA graph",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def _exact_cutlass_toolchain(
    provenance: Mapping[str, Any], expected_version: str
) -> dict[str, object]:
    entries = {
        str(entry[0]): entry[1]
        for entry in provenance["toolchain"]
        if isinstance(entry, list) and len(entry) >= 2
    }
    missing_entries = sorted(set(_CUTLASS_COMPONENTS) - set(entries))
    if missing_entries:
        raise RuntimeError(
            f"manifest omits CUTLASS toolchain entries: {missing_entries}"
        )
    exact = {
        distribution: entries[manifest_name]
        for manifest_name, distribution in _CUTLASS_COMPONENTS.items()
    }
    if exact["nvidia-cutlass-dsl"] != expected_version:
        raise RuntimeError(
            "CUTLASS DSL version mismatch: "
            f"{exact['nvidia-cutlass-dsl']} != {expected_version}"
        )
    invalid = {
        name: version
        for name, version in exact.items()
        if version not in {expected_version, "missing"}
    }
    if invalid:
        raise RuntimeError(
            f"mixed CUTLASS package versions for {expected_version}: {invalid}"
        )
    return exact


def _load_checked(
    cache: Path,
    spec_hash: str,
    *,
    fingerprint: str,
    expected_cutlass: str,
    cache_key: str | None = None,
) -> tuple[object, dict[str, Any]]:
    compiled, loaded = load_exact(cache, spec_hash, cache_key=cache_key)
    if loaded["package_fingerprint"] != fingerprint:
        raise RuntimeError(
            "package fingerprint mismatch: "
            f"{loaded['package_fingerprint']} != {fingerprint}"
        )
    provenance = dict(loaded)
    manifest = json.loads(
        Path(str(loaded["manifest_path"])).read_text(encoding="utf-8")
    )
    provenance["exact_cutlass_toolchain"] = _exact_cutlass_toolchain(
        loaded, expected_cutlass
    )
    provenance["expected_cutlass_dsl"] = expected_cutlass
    provenance["artifact_source_identity"] = {
        "package_fingerprint": loaded["package_fingerprint"],
        "target": manifest.get("target"),
        "target_identity": manifest.get("target_identity"),
        "semantic_key": manifest.get("semantic_key"),
        "compile_environment": manifest.get("compile_environment"),
        "manifest_sha256": loaded["manifest_sha256"],
        "object_sha256": loaded["object_sha256"],
    }
    return compiled, provenance


def _verified_artifact(provenance: Mapping[str, Any]) -> dict[str, object]:
    return {"status": "verified", **verify_artifact(provenance)}


def _mode_snapshot(expected_physical_gpu: int) -> dict[str, object]:
    snapshot = nvidia_smi_gpu_mode_snapshot()
    if not snapshot.get("available"):
        raise RuntimeError(f"nvidia-smi GPU-mode snapshot unavailable: {snapshot}")
    fields_map = snapshot.get("fields")
    if not isinstance(fields_map, dict):
        raise RuntimeError(f"GPU-mode snapshot has no field mapping: {snapshot}")
    if str(fields_map.get("index")) != str(expected_physical_gpu):
        raise RuntimeError(
            "GPU-mode snapshot selected the wrong physical GPU: "
            f"{fields_map.get('index')} != {expected_physical_gpu}"
        )
    if fields_map.get("uuid") != snapshot.get("nvidia_smi_uuid"):
        raise RuntimeError(f"GPU-mode snapshot UUID mismatch: {snapshot}")
    return snapshot


@contextmanager
def _resolution_frozen() -> Iterator[None]:
    already_frozen = kernel_resolution_frozen()
    if not already_frozen:
        freeze_kernel_resolution("exact-object W4A16 serving ABBA")
    try:
        yield
    finally:
        if not already_frozen:
            unfreeze_kernel_resolution()


def _case_kind(m: int) -> str:
    if m <= 6:
        return "direct"
    if m < 1024:
        return "routed"
    return "prefill"


def _specialization(m: int, direct: bool) -> str:
    if direct:
        return "direct"
    return "routed" if m < 1024 else "prefill"


def _make_source_weights() -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(20260718)
    base_w13 = torch.randint(
        0,
        256,
        (1, _INTERMEDIATE, _HIDDEN // 2),
        dtype=torch.uint8,
        device="cuda",
        generator=generator,
    )
    base_w2 = torch.randint(
        0,
        256,
        (1, _HIDDEN, _INTERMEDIATE // 2),
        dtype=torch.uint8,
        device="cuda",
        generator=generator,
    )
    w13 = base_w13.expand(_EXPERTS, -1, -1).contiguous()
    w2 = base_w2.expand(_EXPERTS, -1, -1).contiguous()
    base_w13_scale = (
        torch.rand(
            (1, _INTERMEDIATE, _HIDDEN // 16),
            dtype=torch.float32,
            device="cuda",
            generator=generator,
        )
        * 0.0625
        + 0.03125
    ).to(torch.float8_e4m3fn)
    base_w2_scale = (
        torch.rand(
            (1, _HIDDEN, _INTERMEDIATE // 16),
            dtype=torch.float32,
            device="cuda",
            generator=generator,
        )
        * 0.0625
        + 0.03125
    ).to(torch.float8_e4m3fn)
    w13_scale = swizzle_block_scale(
        base_w13_scale.expand(_EXPERTS, -1, -1).contiguous()
    )
    w2_scale = swizzle_block_scale(base_w2_scale.expand(_EXPERTS, -1, -1).contiguous())
    w13_global = torch.full((_EXPERTS,), 0.0625, dtype=torch.float32, device="cuda")
    w2_global = torch.full((_EXPERTS,), 0.0625, dtype=torch.float32, device="cuda")
    return w13, w13_scale, w13_global, w2, w2_scale, w2_global


def _make_inputs(
    m: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(20260719)
    prototype = (
        torch.randn(
            (1, _HIDDEN),
            dtype=torch.float32,
            device="cuda",
            generator=generator,
        )
        * 0.125
    ).to(torch.bfloat16)
    x = prototype.expand(m, -1).clone()
    token = torch.arange(m, dtype=torch.int64, device="cuda")[:, None]
    offsets = torch.arange(0, _TOPK * 23, 23, dtype=torch.int64, device="cuda")[None, :]
    topk_ids = ((token + offsets) % _EXPERTS).to(torch.int32)
    prototype_weights = torch.tensor(
        (0.25, 0.25, 0.125, 0.125, 0.125, 0.125),
        dtype=torch.float32,
        device="cuda",
    )[None, :]
    topk_weights = prototype_weights.expand(m, -1).contiguous()
    return x, topk_ids, topk_weights, prototype


def _oracle(prototype: torch.Tensor, source: tuple[torch.Tensor, ...]) -> torch.Tensor:
    w13, w13_scale, w13_global, w2, w2_scale, w2_global = source
    return moe_reference_w4a16(
        prototype,
        w13[:1],
        w13_scale[:1],
        w13_global[:1],
        w2[:1],
        w2_scale[:1],
        w2_global[:1],
        torch.zeros((1, 1), dtype=torch.int32, device="cuda"),
        torch.ones((1, 1), dtype=torch.float32, device="cuda"),
        1,
        _HIDDEN,
        _INTERMEDIATE,
        activation=_ACTIVATION,
    )


def _correctness(actual: torch.Tensor, expected_row: torch.Tensor) -> dict[str, object]:
    finite = bool(torch.isfinite(actual).all())
    nonzero = int(torch.count_nonzero(actual))
    expected_f32 = expected_row.float()
    expected_norm = expected_f32.norm()
    min_cos = 1.0
    max_rel_l2 = 0.0
    max_abs = 0.0
    for start in range(0, actual.shape[0], 1024):
        value = actual[start : start + 1024].float()
        expected = expected_f32.expand_as(value)
        diff = value - expected
        cosine = (value * expected).sum(1) / (
            value.norm(dim=1) * expected_norm
        ).clamp_min(1e-24)
        relative = diff.norm(dim=1) / expected_norm
        min_cos = min(min_cos, float(cosine.min()))
        max_rel_l2 = max(max_rel_l2, float(relative.max()))
        max_abs = max(max_abs, float(diff.abs().max()))
    passed = finite and nonzero > 0 and min_cos >= 0.99 and max_rel_l2 <= 0.15
    if not passed:
        raise AssertionError(
            f"W4A16 oracle failed: finite={finite}, nonzero={nonzero}, "
            f"min_cos={min_cos}, max_rel_l2={max_rel_l2}, max_abs={max_abs}"
        )
    return {
        "passed": True,
        "finite": finite,
        "nonzero": nonzero,
        "max_abs": max_abs,
        "cosine": min_cos,
        "min_row_cos": min_cos,
        "max_row_relative_l2": max_rel_l2,
    }


def _fused_launch_from_manifest(
    compiled: object,
    provenance: Mapping[str, Any],
    *,
    size_m: int,
    max_m_blocks: int,
) -> w4a16_kernel.W4A16FusedMoeCompileResult:
    spec = json.loads(str(provenance["compile_spec_json"]))
    facts = spec.get("facts")
    if (
        spec.get("kernel") != "moe.w4a16.fused_moe"
        or not isinstance(facts, list)
        or len(facts) != 3
        or facts[0] != "w4a16_fused_moe"
        or not isinstance(facts[2], list)
    ):
        raise RuntimeError(f"unexpected fused compile specification: {spec}")
    key = facts[2]
    if len(key) < 27 or not isinstance(key[21], list) or not isinstance(key[22], list):
        raise RuntimeError(f"incomplete fused compile specification facts: {key}")
    fc1 = key[21]
    fc2 = key[22]
    if len(fc1) < 27 or len(fc2) < 27:
        raise RuntimeError("incomplete fused FC1/FC2 specification facts")
    return w4a16_kernel.W4A16FusedMoeCompileResult(
        compiled=compiled,
        size_m=size_m,
        hidden_size=int(key[0]),
        intermediate_size=int(key[1]),
        num_experts=int(key[3]),
        top_k=int(key[4]),
        activation=str(key[5]),
        apply_router_weight_on_input=bool(key[14]),
        zero_fc2_output=bool(key[15]),
        element_dtype=str(key[16]),
        fast_math=bool(key[17]),
        swiglu_limit=float(key[9]) if bool(key[8]) else None,
        swiglu_alpha=float(key[10]),
        swiglu_beta=float(key[11]),
        fc1_tile_n=int(fc1[9]),
        fc1_tile_k=int(fc1[10]),
        fc2_tile_n=int(fc2[9]),
        fc2_tile_k=int(fc2[10]),
        moe_block_size=int(fc1[12]),
        max_m_blocks=max_m_blocks,
        blocks_per_sm=int(key[26]),
        weight_layout=str(key[12]),
        w13_layout=str(fc1[17]),
        direct_topk_routes=bool(key[18]),
        scale_format=str(key[13]),
        tc_decode_fused_sum=bool(fc2[21]),
        collect_activation_amax=bool(key[20]),
    )


@contextmanager
def _pin_exact_dispatch(
    fused: w4a16_kernel.W4A16FusedMoeCompileResult,
    topk_sum: w4a16_kernel.W4A16TopKSumCompileResult,
    records: list[dict[str, object]],
) -> Iterator[None]:
    original_fused = w4a16_kernel.compile_w4a16_fused_moe
    original_sum = w4a16_kernel.compile_w4a16_topk_sum
    original_small_direct = w4a16_kernel._small_m_direct_supported

    def pinned_fused(**kwargs):
        records.append({"kind": "fused", "kwargs": dict(kwargs)})
        return fused

    def pinned_sum(**kwargs):
        records.append({"kind": "topk_sum", "kwargs": dict(kwargs)})
        return topk_sum

    w4a16_kernel.compile_w4a16_fused_moe = pinned_fused
    w4a16_kernel.compile_w4a16_topk_sum = pinned_sum
    w4a16_kernel._small_m_direct_supported = lambda **_kwargs: False
    try:
        yield
    finally:
        w4a16_kernel.compile_w4a16_fused_moe = original_fused
        w4a16_kernel.compile_w4a16_topk_sum = original_sum
        w4a16_kernel._small_m_direct_supported = original_small_direct


def _capture(
    launch,
    *,
    fused: w4a16_kernel.W4A16FusedMoeCompileResult,
    topk_sum: w4a16_kernel.W4A16TopKSumCompileResult,
    stream: torch.cuda.Stream,
    records: list[dict[str, object]],
    debug_path: Path | None,
) -> torch.cuda.CUDAGraph:
    with _pin_exact_dispatch(fused, topk_sum, records):
        with torch.cuda.stream(stream):
            launch()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        if debug_path is not None:
            graph.enable_debug_mode()
        with torch.cuda.graph(graph, stream=stream):
            launch()
        stream.synchronize()
        if debug_path is not None:
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            graph.debug_dump(str(debug_path))
    kinds = {str(record["kind"]) for record in records}
    if kinds != {"fused", "topk_sum"}:
        raise AssertionError(
            f"production graph did not resolve both exact objects: {records}"
        )
    return graph


def _tensor_tree_sha256(value: object) -> dict[str, str]:
    result: dict[str, str] = {}
    seen_containers: set[int] = set()
    seen_tensors: set[tuple[int, int, str]] = set()

    def visit(item: object, name: str) -> None:
        if isinstance(item, torch.Tensor):
            identity = (item.data_ptr(), item.numel(), str(item.dtype))
            if identity not in seen_tensors:
                seen_tensors.add(identity)
                result[name] = tensor_sha256(item)
            return
        if item is None or isinstance(item, (str, int, float, bool)):
            return
        identity = id(item)
        if identity in seen_containers:
            return
        seen_containers.add(identity)
        if is_dataclass(item):
            for field in fields(item):
                if field.name == "workspace":
                    continue
                visit(getattr(item, field.name), f"{name}.{field.name}")
        elif isinstance(item, Mapping):
            for key in sorted(item, key=str):
                visit(item[key], f"{name}.{key}")
        elif isinstance(item, (tuple, list)):
            for index, child in enumerate(item):
                visit(child, f"{name}.{index}")

    visit(value, "prepared")
    if not result:
        raise RuntimeError("prepared W4A16 state exposed no immutable tensors")
    return result


def _pointer_snapshot(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    buffers,
    prepared,
) -> dict[str, dict[str, object]]:
    tensors = {
        "x": x,
        "topk_ids": topk_ids,
        "topk_weights": topk_weights,
        "output": buffers.output,
        "intermediate_cache13": buffers.intermediate_cache13,
        "intermediate_cache2": buffers.intermediate_cache2,
        "fc1_c_tmp": buffers.fc1_c_tmp,
        "fc2_c_tmp": buffers.fc2_c_tmp,
        "packed_route_indices": buffers.packed_route_indices,
        "block_expert_ids": buffers.block_expert_ids,
        "packed_route_count": buffers.packed_route_count,
        "expert_offsets": buffers.expert_offsets,
        "workspace": prepared.workspace,
    }
    return {
        name: {
            "address": tensor.data_ptr(),
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "capacity_bytes": tensor.numel() * tensor.element_size(),
        }
        for name, tensor in tensors.items()
    }


def _plan_record(plan: object) -> dict[str, object]:
    if not is_dataclass(plan):
        raise RuntimeError("W4A16 buffer plan is not a dataclass")
    return {
        field.name: getattr(plan, field.name)
        for field in fields(plan)
        if isinstance(getattr(plan, field.name), (str, int, float, bool, type(None)))
    }


def _run_case(
    m: int,
    *,
    source: tuple[torch.Tensor, ...],
    prepared,
    read_only_sha256_before: Mapping[str, str],
    caches: Mapping[str, Path],
    fingerprints: Mapping[str, str],
    cutlass_versions: Mapping[str, str],
    labels: tuple[str, str],
    force_routed: Mapping[str, bool],
    l2_flush_bytes: int,
    args: argparse.Namespace,
) -> dict[str, object]:
    kind = _case_kind(m)
    x, topk_ids, topk_weights, prototype = _make_inputs(m)
    buffers = make_w4a16_packed_buffers(
        prepared,
        m=m,
        topk=_TOPK,
        dtype=torch.bfloat16,
        device="cuda",
    )
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    sms = int(props.multi_processor_count)
    block_size = select_route_block_size_m(m, _TOPK, _EXPERTS)
    plan = plan_w4a16_buffers(
        prepared,
        m=m,
        topk=_TOPK,
        route_num_experts=_EXPERTS,
        sms=sms,
    )
    direct_by_label = {label: m <= 6 and not force_routed[label] for label in labels}
    if direct_by_label[labels[0]] != direct_by_label[labels[1]]:
        raise ValueError("both arms must use the same serving route policy")

    fused_objects: dict[str, object] = {}
    fused_provenance: dict[str, dict[str, Any]] = {}
    for label in labels:
        specialization = _specialization(m, direct_by_label[label])
        fused_objects[label], fused_provenance[label] = _load_checked(
            caches[label],
            _FUSED_SPEC_HASHES[specialization],
            fingerprint=fingerprints[label],
            expected_cutlass=cutlass_versions[label],
        )
        if fused_provenance[label]["kernel_id"] != "moe.w4a16.fused_moe":
            raise RuntimeError("fused manifest has the wrong kernel id")

    topk_objects: dict[str, object] = {}
    topk_provenance: dict[str, dict[str, Any]] = {}
    shared_topk = args.shared_topk_cache is not None
    if shared_topk:
        shared_object, shared_provenance = _load_checked(
            args.shared_topk_cache,
            _TOPK_SPEC_HASH,
            cache_key=args.shared_topk_cache_key,
            fingerprint=args.shared_topk_fingerprint,
            expected_cutlass=args.shared_topk_cutlass_version,
        )
        for label in labels:
            topk_objects[label] = shared_object
            topk_provenance[label] = dict(shared_provenance)
    else:
        cache_keys = {
            labels[0]: args.a_topk_cache_key,
            labels[1]: args.b_topk_cache_key,
        }
        for label in labels:
            topk_objects[label], topk_provenance[label] = _load_checked(
                caches[label],
                _TOPK_SPEC_HASH,
                cache_key=cache_keys[label],
                fingerprint=fingerprints[label],
                expected_cutlass=cutlass_versions[label],
            )
    for label in labels:
        if topk_provenance[label]["kernel_id"] != "moe.w4a16.topk_sum":
            raise RuntimeError("top-k-sum manifest has the wrong kernel id")
    if (
        fused_provenance[labels[0]]["compile_spec_json"]
        != fused_provenance[labels[1]]["compile_spec_json"]
        or topk_provenance[labels[0]]["compile_spec_json"]
        != topk_provenance[labels[1]]["compile_spec_json"]
    ):
        raise AssertionError("arm compile specifications differ")

    launches: dict[
        str,
        tuple[
            w4a16_kernel.W4A16FusedMoeCompileResult,
            w4a16_kernel.W4A16TopKSumCompileResult,
        ],
    ] = {}
    provenance: dict[str, dict[str, object]] = {}
    for label in labels:
        direct = direct_by_label[label]
        max_m_blocks = m * _TOPK if direct else int(plan.route_blocks)
        fused = _fused_launch_from_manifest(
            fused_objects[label],
            fused_provenance[label],
            size_m=m,
            max_m_blocks=max_m_blocks,
        )
        if (
            fused.hidden_size != _HIDDEN
            or fused.intermediate_size != _INTERMEDIATE
            or fused.num_experts != _EXPERTS
            or fused.top_k != _TOPK
            or fused.activation != _ACTIVATION
            or fused.moe_block_size != block_size
            or fused.direct_topk_routes != direct
            or fused.tc_decode_fused_sum
        ):
            raise RuntimeError(
                f"{label}: exact fused object metadata mismatch: {fused}"
            )
        topk_sum = w4a16_kernel.W4A16TopKSumCompileResult(
            compiled=topk_objects[label],
            m=0,
            topk=_TOPK,
            hidden_size=_HIDDEN,
        )
        launches[label] = (fused, topk_sum)
        provenance[label] = {
            "policy": _specialization(m, direct),
            "fused": fused_provenance[label],
            "topk_sum": topk_provenance[label],
        }

    artifacts_before = {
        label: {
            component: _verified_artifact(component_provenance)
            for component, component_provenance in (
                ("fused", fused_provenance[label]),
                ("topk_sum", topk_provenance[label]),
            )
        }
        for label in labels
    }
    input_hashes_before = {
        "x": tensor_sha256(x),
        "topk_ids": tensor_sha256(topk_ids),
        "topk_weights": tensor_sha256(topk_weights),
    }
    pointers_before = _pointer_snapshot(x, topk_ids, topk_weights, buffers, prepared)

    def launch(fused, topk_sum):
        return w4a16_kernel.run_w4a16_moe(
            x,
            prepared,
            topk_weights,
            topk_ids,
            activation=_ACTIVATION,
            fast_math=True,
            intermediate_cache13=buffers.intermediate_cache13,
            intermediate_cache2=buffers.intermediate_cache2,
            output=buffers.output,
            fc1_c_tmp=buffers.fc1_c_tmp,
            fc2_c_tmp=buffers.fc2_c_tmp,
            packed_route_indices=buffers.packed_route_indices,
            block_expert_ids=buffers.block_expert_ids,
            packed_route_count=buffers.packed_route_count,
            expert_offsets=buffers.expert_offsets,
            fused_launch=fused,
            topk_sum_launch=topk_sum,
        )

    stream = torch.cuda.Stream()
    graphs: dict[str, torch.cuda.CUDAGraph] = {}
    topologies: dict[str, dict[str, object]] = {}
    dispatch_records: dict[str, list[dict[str, object]]] = {}
    compile_misses_before_capture = int(
        cute_compiler.compile_cache_info()["compile_misses"]
    )
    for label in labels:
        fused, topk_sum = launches[label]
        debug_path = None
        if args.graph_dump_dir is not None:
            safe_label = label.replace("/", "_")
            debug_path = args.graph_dump_dir / f"m{m}-{safe_label}.dot"
        records: list[dict[str, object]] = []
        graphs[label] = _capture(
            lambda fused=fused, topk_sum=topk_sum: launch(fused, topk_sum),
            fused=fused,
            topk_sum=topk_sum,
            stream=stream,
            records=records,
            debug_path=debug_path,
        )
        topologies[label] = graph_topology(graphs[label])
        dispatch_records[label] = records
    compile_misses_after_capture = int(
        cute_compiler.compile_cache_info()["compile_misses"]
    )
    if compile_misses_after_capture != compile_misses_before_capture:
        raise AssertionError("serving graph capture triggered CUTLASS compilation")
    topology_equal = topology_signature(topologies[labels[0]]) == topology_signature(
        topologies[labels[1]]
    )
    if not topology_equal:
        raise AssertionError("serving arm CUDA graph topologies differ")
    if direct_by_label[labels[0]]:
        minimum_kernel_nodes = 2
    elif kind == "prefill":
        minimum_kernel_nodes = 5
    else:
        minimum_kernel_nodes = 4
    for label in labels:
        if int(topologies[label]["kernel_node_count"]) < minimum_kernel_nodes:
            raise AssertionError(
                f"{label}: graph bypassed exact fused/top-k serving path: "
                f"{topologies[label]}"
            )

    replay_allocator_records: list[dict[str, object]] = []

    def replay_checked(
        label: str,
        expected: torch.Tensor,
        *,
        phase: str,
    ) -> tuple[dict[str, object], torch.Tensor]:
        with torch.cuda.stream(stream):
            buffers.output.fill_(float("nan"))
        stream.synchronize()
        allocator_before = allocator_counters()
        with torch.cuda.stream(stream):
            graphs[label].replay()
        stream.synchronize()
        allocator_after = allocator_counters()
        if allocator_after != allocator_before:
            raise AssertionError(
                f"{label}: CUDA allocator changed during graph replay: "
                f"{allocator_before} != {allocator_after}"
            )
        if bool(torch.isnan(buffers.output).any()):
            raise AssertionError(f"{label} did not overwrite poisoned output")
        replay_allocator_records.append(
            {
                "phase": phase,
                "label": label,
                "before": allocator_before,
                "after": allocator_after,
            }
        )
        return _correctness(buffers.output, expected), buffers.output.clone()

    baseline_expected = _oracle(prototype, source)
    correctness: dict[str, dict[str, dict[str, object]]] = {
        "baseline": {},
        "mutated_live_input": {},
        "post_timing": {},
    }
    baseline_outputs: dict[str, torch.Tensor] = {}
    for label in labels:
        correctness["baseline"][label], baseline_outputs[label] = replay_checked(
            label, baseline_expected, phase="scenario_0"
        )
    if not torch.equal(baseline_outputs[labels[0]], baseline_outputs[labels[1]]):
        raise AssertionError("W4A16 serving arms are not bit exact")
    scenario_0_output_sha256 = {
        label: tensor_sha256(baseline_outputs[label]) for label in labels
    }

    x.neg_()
    topk_ids.add_(1).remainder_(_EXPERTS)
    topk_weights.mul_(0.5)
    scenario_1_input_sha256 = {
        "x": tensor_sha256(x),
        "topk_ids": tensor_sha256(topk_ids),
        "topk_weights": tensor_sha256(topk_weights),
    }
    changed_inputs = {
        name: scenario_1_input_sha256[name] != input_hashes_before[name]
        for name in input_hashes_before
    }
    if not all(changed_inputs.values()):
        raise AssertionError(
            f"scenario-1 did not mutate every captured serving input: {changed_inputs}"
        )
    pointers_scenario_1 = _pointer_snapshot(
        x, topk_ids, topk_weights, buffers, prepared
    )
    if pointers_scenario_1 != pointers_before:
        raise AssertionError("live serving inputs were not mutated in place")
    mutated_expected = _oracle(-prototype, source) * 0.5
    mutated_outputs: dict[str, torch.Tensor] = {}
    for label in labels:
        correctness["mutated_live_input"][label], mutated_outputs[label] = (
            replay_checked(label, mutated_expected, phase="scenario_1")
        )
    if not torch.equal(mutated_outputs[labels[0]], mutated_outputs[labels[1]]):
        raise AssertionError("mutated-input W4A16 serving arms are not bit exact")
    scenario_1_output_sha256 = {
        label: tensor_sha256(mutated_outputs[label]) for label in labels
    }
    changed_outputs = {
        label: scenario_1_output_sha256[label] != scenario_0_output_sha256[label]
        for label in labels
    }
    if not all(changed_outputs.values()):
        raise AssertionError(
            f"live serving input mutation did not change every arm: {changed_outputs}"
        )
    x.neg_()
    topk_ids.sub_(1).remainder_(_EXPERTS)
    topk_weights.mul_(2.0)
    restored_hashes = {
        "x": tensor_sha256(x),
        "topk_ids": tensor_sha256(topk_ids),
        "topk_weights": tensor_sha256(topk_weights),
    }
    if restored_hashes != input_hashes_before:
        raise AssertionError("live serving inputs were not restored before timing")
    del baseline_outputs, mutated_outputs

    timed_input_hashes_before = dict(restored_hashes)
    case_cycles = args.prefill_cycles if kind == "prefill" else args.cycles
    conditions = time_conditions(
        graphs,
        labels=labels,
        precondition=args.precondition,
        warmup=args.warmup,
        cycles=case_cycles,
        event_batch_cycles=args.event_batch_cycles,
        stream=stream,
        cold_l2=True,
        l2_flush_bytes=l2_flush_bytes,
        precondition_seconds=args.precondition_seconds,
        maximum_precondition_seconds=args.maximum_precondition_seconds,
        mode_snapshot=lambda: _mode_snapshot(int(args.expected_physical_gpu)),
        required_pstate="P1",
        required_active_throttle_reasons=(args.required_active_throttle_reasons),
        max_sm_clock_delta_mhz=args.max_sm_clock_delta_mhz,
        allow_sw_power_cap_transition=args.allow_sw_power_cap_transition,
    )
    timed_input_hashes_after = {
        "x": tensor_sha256(x),
        "topk_ids": tensor_sha256(topk_ids),
        "topk_weights": tensor_sha256(topk_weights),
    }
    if timed_input_hashes_after != timed_input_hashes_before:
        raise AssertionError("scenario-0 serving inputs changed during timed replay")
    compile_spec_hashes = {
        label: {
            "fused": str(fused_provenance[label]["compile_spec_hash"]),
            "topk_sum": str(topk_provenance[label]["compile_spec_hash"]),
        }
        for label in labels
    }
    all_graph_spec_hashes = sorted(
        {
            spec_hash
            for by_component in compile_spec_hashes.values()
            for spec_hash in by_component.values()
        }
    )
    for condition in conditions.values():
        condition["compile_spec_hashes"] = compile_spec_hashes
        condition["all_graph_spec_hashes"] = all_graph_spec_hashes
        for label in labels:
            count = int(condition["timings"]["summaries"][label]["count"])
            if count < 1000:
                raise AssertionError(f"{label}: only {count} raw timing samples")

    post_timing_output_sha256: dict[str, str] = {}
    for label in labels:
        metric, post_timing_output = replay_checked(
            label, baseline_expected, phase="post_timing"
        )
        correctness["post_timing"][label] = metric
        post_timing_output_sha256[label] = tensor_sha256(post_timing_output)
        del post_timing_output
    input_hashes_after = {
        "x": tensor_sha256(x),
        "topk_ids": tensor_sha256(topk_ids),
        "topk_weights": tensor_sha256(topk_weights),
    }
    if input_hashes_after != input_hashes_before:
        raise AssertionError("W4A16 serving inputs changed during timed replay")
    pointers_after = _pointer_snapshot(x, topk_ids, topk_weights, buffers, prepared)
    if pointers_after != pointers_before:
        raise AssertionError("W4A16 serving pointers/capacities changed")
    artifacts_after = {
        label: {
            component: _verified_artifact(component_provenance)
            for component, component_provenance in (
                ("fused", fused_provenance[label]),
                ("topk_sum", topk_provenance[label]),
            )
        }
        for label in labels
    }
    if artifacts_after != artifacts_before:
        raise AssertionError("W4A16 cached artifacts changed during benchmark")
    allocator_checks = list(replay_allocator_records)
    allocator_checks.extend(
        {
            "phase": f"timing_{condition_name}",
            "before": condition["allocator_before"],
            "after": condition["allocator_after"],
        }
        for condition_name, condition in conditions.items()
    )

    return {
        "shape": {
            "m": m,
            "hidden_size": _HIDDEN,
            "intermediate_size": _INTERMEDIATE,
            "experts": _EXPERTS,
            "topk": _TOPK,
        },
        "serving_regime": kind,
        "arm_policies": {
            label: {
                "direct_topk_routes": direct_by_label[label],
                "production_route_pack_in_graph": not direct_by_label[label],
                "small_m_direct_bypass_disabled": True,
            }
            for label in labels
        },
        "shared_topk_object": shared_topk,
        "block_size": block_size,
        "fixed_workspace": _plan_record(plan),
        "stable_tensor_pointers": pointers_before,
        "same_addresses": {
            name: record["address"] for name, record in pointers_before.items()
        },
        "same_address_arms": True,
        "same_address_across_arms": True,
        "graph_reuse": {
            "captured_graph_reused": True,
            "in_place": True,
            "same_addresses": True,
            "allocation_addresses_stable": True,
            "addresses_before": pointers_before,
            "addresses_scenario_1": pointers_scenario_1,
            "addresses_after": pointers_after,
        },
        "fixed_workspace_capacity": True,
        "fixed_allocation": True,
        "cuda_graph_replay": True,
        "cuda_graph_topology": topologies,
        "cuda_graph_topology_equal": True,
        "production_resolution": dispatch_records,
        "production_topk_sum_in_graph": True,
        "allocator_replay_records": replay_allocator_records,
        "allocator_checks": allocator_checks,
        "allocator_stable": True,
        "zero_replay_allocations": True,
        "input_sha256": input_hashes_before,
        "input_immutable": True,
        "read_only_inputs_unchanged": True,
        "read_only_inputs_immutable": True,
        "read_only_inputs": {
            "unchanged": True,
            "sha256_before": dict(read_only_sha256_before),
            "sha256_after": dict(read_only_sha256_before),
            "timed_live_scenario_0": {
                "unchanged": True,
                "sha256_before": timed_input_hashes_before,
                "sha256_after": timed_input_hashes_after,
            },
        },
        "poisoned_outputs_overwritten": True,
        "poisoned_output_overwritten": True,
        "arms_bit_exact": True,
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
            "scenario_0_sha256": input_hashes_before,
            "scenario_1_sha256": scenario_1_input_sha256,
            "scenario_0_output_sha256": scenario_0_output_sha256,
            "scenario_1_output_sha256": scenario_1_output_sha256,
        },
        "post_timing_output_sha256": post_timing_output_sha256,
        "correctness": correctness,
        "oracle_metrics": correctness["baseline"],
        "precondition": args.precondition,
        "warmup": args.warmup,
        "cycles": case_cycles,
        "conditions": conditions,
        "compile_spec_hashes": compile_spec_hashes,
        "all_graph_spec_hashes": all_graph_spec_hashes,
        "object_provenance": provenance,
        "artifact_verification_before": artifacts_before,
        "artifact_verification_after": artifacts_after,
    }


def main() -> None:
    args = _args()
    gpu = require_target_gpu(args.expected_physical_gpu)
    labels = (args.a_label, args.b_label)
    if labels[0] == labels[1]:
        raise ValueError("arm labels must differ")
    required_matrix = {
        "full": _REQUIRED_M,
        "decode": _REQUIRED_DECODE_M,
        "prefill": _REQUIRED_PREFILL_M,
    }[args.matrix_scope]
    matrix = tuple(sorted(set(args.m or required_matrix)))
    if args.m is not None and len(args.m) != len(set(args.m)):
        raise ValueError("duplicate --m values are not allowed")
    if matrix != required_matrix:
        raise ValueError(
            f"--m must cover the exact {args.matrix_scope} matrix "
            f"{required_matrix}; got {matrix}"
        )
    if args.a_force_routed != args.b_force_routed:
        raise ValueError("both arms must use the same route policy")
    if args.cycles < 500 or args.prefill_cycles < 500:
        raise ValueError("decode and prefill cycles must each be at least 500")
    if args.precondition < 1 or args.warmup < 1 or args.event_batch_cycles < 1:
        raise ValueError("precondition, warmup, and event batch must be positive")
    if args.precondition_seconds < 5.0:
        raise ValueError("precondition seconds must be at least 5")
    if not (args.precondition_seconds <= args.maximum_precondition_seconds <= 60.0):
        raise ValueError("maximum precondition seconds must cover the minimum and <=60")
    if not 0.0 < args.max_sm_clock_delta_mhz <= 60.0:
        raise ValueError("maximum SM clock delta must be in (0, 60]")
    mode_policy = timing_mode_policy(
        required_pstate="P1",
        required_active_throttle_reasons=args.required_active_throttle_reasons,
        max_sm_clock_delta_mhz=args.max_sm_clock_delta_mhz,
        allow_sw_power_cap_transition=args.allow_sw_power_cap_transition,
    )
    if not args.cold_l2:
        raise ValueError("warm-L2 and cold-L2 conditions are both required")
    l2_flush_bytes = resolve_l2_flush_bytes(args.l2_flush_bytes)
    if l2_flush_bytes <= 0:
        raise RuntimeError("cold-L2 evidence requires a positive flush capacity")
    shared_values = (
        args.shared_topk_cache_key,
        args.shared_topk_fingerprint,
        args.shared_topk_cutlass_version,
    )
    if args.shared_topk_cache is None and any(
        value is not None for value in shared_values
    ):
        raise ValueError("shared top-k metadata requires --shared-topk-cache")
    if args.shared_topk_cache is not None and (
        args.shared_topk_fingerprint is None or args.shared_topk_cutlass_version is None
    ):
        raise ValueError("shared top-k mode requires fingerprint and CUTLASS version")

    caches = {labels[0]: args.a_cache, labels[1]: args.b_cache}
    fingerprints = {
        labels[0]: args.a_fingerprint,
        labels[1]: args.b_fingerprint,
    }
    cutlass_versions = {
        labels[0]: args.a_cutlass_version,
        labels[1]: args.b_cutlass_version,
    }
    force_routed = {
        labels[0]: args.a_force_routed,
        labels[1]: args.b_force_routed,
    }
    initial_mode = _mode_snapshot(int(args.expected_physical_gpu))
    compile_cache_initial = cute_compiler.compile_cache_info()

    source = _make_source_weights()
    prepared = prepare_w4a16_modelopt_nvfp4_weights(
        *source,
        activation=_ACTIVATION,
        params_dtype=torch.bfloat16,
    )
    torch.cuda.synchronize()
    prepared_hashes_before = _tensor_tree_sha256(prepared)
    with _resolution_frozen():
        cases = [
            _run_case(
                m,
                source=source,
                prepared=prepared,
                read_only_sha256_before=prepared_hashes_before,
                caches=caches,
                fingerprints=fingerprints,
                cutlass_versions=cutlass_versions,
                labels=labels,
                force_routed=force_routed,
                l2_flush_bytes=l2_flush_bytes,
                args=args,
            )
            for m in matrix
        ]
    prepared_hashes_after = _tensor_tree_sha256(prepared)
    if prepared_hashes_after != prepared_hashes_before:
        raise AssertionError("prepared W4A16 read-only weights/scales changed")
    for case in cases:
        read_only_inputs = case.get("read_only_inputs")
        if not isinstance(read_only_inputs, dict):
            raise AssertionError("W4A16 case omitted read-only input evidence")
        read_only_inputs["sha256_after"] = dict(prepared_hashes_after)
        read_only_inputs["unchanged"] = True
    compile_cache_final = cute_compiler.compile_cache_info()
    if int(compile_cache_final["compile_misses"]) != int(
        compile_cache_initial["compile_misses"]
    ):
        raise AssertionError("W4A16 serving ABBA triggered CUTLASS compilation")
    final_mode = _mode_snapshot(int(args.expected_physical_gpu))
    if int(final_mode["captured_unix_ns"]) <= int(initial_mode["captured_unix_ns"]):
        raise AssertionError("GPU-mode snapshots are not ordered")

    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    source_paths = {
        "benchmark": Path(__file__).resolve(),
        "kernel": REPO_ROOT / "b12x/moe/fused/w4a16/kernel.py",
        "host": REPO_ROOT / "b12x/moe/fused/w4a16/host.py",
        "prepare": REPO_ROOT / "b12x/moe/fused/w4a16/prepare.py",
        "compiler": REPO_ROOT / "b12x/cute/compiler.py",
        "exact_cache_abba": CORE_ROOT / "exact_cache_abba.py",
        "gpu_scope": CORE_ROOT / "gpu_scope.py",
    }
    result = {
        "schema": "b12x.w4a16.serving.cache_abba.v2",
        "evidence_status": args.evidence_status,
        "command": [sys.executable, *sys.argv],
        "labels": {"a": labels[0], "b": labels[1]},
        "gpu": {
            **gpu,
            "logical_index": torch.cuda.current_device(),
            "sms": properties.multi_processor_count,
            "capability": list(torch.cuda.get_device_capability()),
        },
        "gpu_mode_initial": initial_mode,
        "gpu_mode_final": final_mode,
        "worktree": str(REPO_ROOT),
        "git_head": _git("rev-parse", "HEAD"),
        "git_status_porcelain": _git("status", "--short"),
        "source_sha256": {
            name: sha256_file(path) for name, path in source_paths.items()
        },
        "host_package_fingerprint": cute_compiler.b12x_package_fingerprint(),
        "caches": {label: str(cache.resolve()) for label, cache in caches.items()},
        "shared_topk_cache": (
            str(args.shared_topk_cache.resolve())
            if args.shared_topk_cache is not None
            else None
        ),
        "required_decode_m": list(_REQUIRED_DECODE_M),
        "required_prefill_m": list(_REQUIRED_PREFILL_M),
        "matrix_scope": args.matrix_scope,
        "matrix_m": list(matrix),
        "balanced_order": [labels[0], labels[1], labels[1], labels[0]],
        "balanced_alternate": [labels[1], labels[0], labels[0], labels[1]],
        "timing_mode_policy": mode_policy,
        "l2_flush_bytes": l2_flush_bytes,
        "prepared_read_only_input_sha256": prepared_hashes_before,
        "read_only_inputs": {
            "unchanged": True,
            "sha256_before": prepared_hashes_before,
            "sha256_after": prepared_hashes_after,
        },
        "no_recompilation": True,
        "compile_cache_initial": compile_cache_initial,
        "compile_cache_final": compile_cache_final,
        "cases": cases,
        "all_correct": True,
        "all_graph_topologies_equal": True,
        "all_same_address_arms": True,
        "all_fixed_workspace_capacity": True,
        "all_allocator_stable": True,
        "all_zero_replay_allocations": True,
        "all_input_immutable": True,
        "all_read_only_inputs_immutable": True,
        "all_poisoned_outputs_overwritten": True,
        "all_arms_bit_exact": True,
        "all_live_input_scenarios_distinct": True,
        "all_live_input_mutations_changed_input": True,
        "all_live_input_mutations_changed_output": True,
        "all_evidence_gates_passed": True,
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
                "matrix_m": list(matrix),
                "shared_topk_object": args.shared_topk_cache is not None,
                "required_active_throttle_reasons": (
                    args.required_active_throttle_reasons
                ),
                "all_evidence_gates_passed": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
