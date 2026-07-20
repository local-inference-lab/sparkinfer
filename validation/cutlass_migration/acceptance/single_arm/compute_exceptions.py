#!/usr/bin/env python3
"""Run one CUTLASS arm of the real compute-exception CUDA-graph corpus.

The case matrix is exactly the production matrix owned by
``validation.cutlass_migration.diagnostics.paired.compute_exceptions``.  Each
process loads only one
arm's immutable cache objects, resolves the normal production launcher to
those objects, captures one fixed-address graph, validates two live-input
scenarios against the paired producer's independent GPU oracle, and measures
warm and cold L2 with the shared hardened single-graph timer.

This producer is GPU-only evidence.  It has no CPU/reference-only acceptance
route and is restricted by the shared process envelope to physical GPU 4 or 5.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch

from validation.cutlass_migration.diagnostics.paired.compute_exceptions import (
    CaseContract,
    CaseState,
    _CASES as PAIRED_CASES,
    _TIMER_LIMITED_CASES,
    _build_state,
    _controlled_dispatch_environment,
    _correctness,
    _exact_compile_resolution,
    _validate_manifest_kernels,
)
from validation.cutlass_migration.core.comparison_identity import (
    comparison_semantic_key_from_manifest,
)
from validation.cutlass_migration.core.exact_cache_abba import (
    allocator_counters,
    exact_artifact_evidence,
    gpu_mode_snapshot,
    graph_topology,
    json_sha256,
    load_exact,
    sha256_file,
    single_graph_topology,
    tensor_sha256,
    time_single_graph_conditions,
    verify_artifact,
)
from validation.cutlass_migration.core.single_arm_e2e import (
    ReviewedCaseBinding,
    add_single_arm_arguments,
    begin_single_arm_session,
    bind_exact_artifact,
    build_exact_launch_plan,
    finish_single_arm_session,
    verify_case_compile_contract,
)
import sparkinfer.cute.compiler as cute_compiler


FAMILY = "compute_exceptions"
INPUT_SCHEMA = "sparkinfer.compute_exceptions.end_to_end_input.v1"
GUARD_ELEMENTS = 256
GUARD_VALUE = 123.0

_ARTIFACT_ROLES = {
    "dense-nvfp4-m32": ("dense-gemm",),
    "dense-fused-quant-m2": ("dense-fused-quant",),
    "dense-grouped-fused-quant-m2-g2": ("dense-grouped-fused-quant",),
    "tp-moe-nvfp4-micro-m2": ("tp-moe-micro-direct",),
    "tp-moe-w4a8-mx-tiny-m2": (
        "tp-moe-tiny-phase1",
        "tp-moe-tiny-phase2",
    ),
    "w4a16-standalone-gemm-m128": ("w4a16-standalone-gemm",),
    "w4a16-native-e8m0-small-m1": ("w4a16-small-m-direct",),
    "w4a16-swiglu-limit-m24": ("w4a16-fused-moe", "w4a16-topk-sum"),
}

_LIVE_TENSOR_NAMES = {
    "dense-nvfp4-m32": ("a_values", "a_scales", "alpha"),
    "dense-fused-quant-m2": ("source",),
    "dense-grouped-fused-quant-m2-g2": ("source",),
    "tp-moe-nvfp4-micro-m2": (
        "activations",
        "topk_ids",
        "topk_weights",
    ),
    "tp-moe-w4a8-mx-tiny-m2": (
        "activations",
        "topk_ids",
        "topk_weights",
    ),
    "w4a16-standalone-gemm-m128": ("x",),
    "w4a16-native-e8m0-small-m1": ("x", "topk_ids", "topk_weights"),
    "w4a16-swiglu-limit-m24": ("x", "topk_ids", "topk_weights"),
}

_QUANTIZATION_GATES = {
    "dense-nvfp4-m32": "nvfp4-quantization-semantics",
    "dense-fused-quant-m2": "mxfp8-fused-quantization-semantics",
    "dense-grouped-fused-quant-m2-g2": ("mxfp8-grouped-fused-quantization-semantics"),
    "tp-moe-nvfp4-micro-m2": "nvfp4-moe-quantization-semantics",
    "tp-moe-w4a8-mx-tiny-m2": "w4a8-mx-quantization-semantics",
    "w4a16-standalone-gemm-m128": "w4a16-modelopt-nvfp4-semantics",
    "w4a16-native-e8m0-small-m1": "w4a16-native-e8m0-semantics",
    "w4a16-swiglu-limit-m24": "w4a16-modelopt-nvfp4-semantics",
}

_SOURCE_DESCRIPTIONS = {
    "dense-nvfp4-m32": {
        "generator_seed": 46_001,
        "scenarios": "two independently quantized NVFP4 activation operands",
    },
    "dense-fused-quant-m2": {
        "generator_seed": 46_002,
        "scenarios": "two BF16 inputs fused-quantized to MXFP8 in production",
    },
    "dense-grouped-fused-quant-m2-g2": {
        "generator_seed": 46_003,
        "scenarios": "two grouped BF16 inputs fused-quantized to MXFP8",
    },
    "tp-moe-nvfp4-micro-m2": {
        "weight_seed": 101,
        "scenario_seeds": [102, 103],
        "route_shifts": [0, 2],
    },
    "tp-moe-w4a8-mx-tiny-m2": {
        "weight_seed": 301,
        "scenario_seeds": [302, 303],
        "route_shifts": [0, 2],
    },
    "w4a16-standalone-gemm-m128": {
        "activation_seed": 20_260_718,
        "scenarios": "two BF16 activation matrices with fixed packed routes",
    },
    "w4a16-native-e8m0-small-m1": {
        "generator_seed": 20_261_602,
        "scenarios": "two native-E8M0 activation/route/weight scenarios",
    },
    "w4a16-swiglu-limit-m24": {
        "weight_seed": 20_260_519,
        "activation_seed": 20_260_719,
        "scenarios": "baseline and sign/route/weight-mutated SwiGLU-limit inputs",
    },
}

_ROUTE_PACK_ROLES = {
    "_pack_topk_routes_small_prefix_kernel": "route-pack-small-prefix",
    "_pack_topk_routes_prefix_kernel": "route-pack-prefix",
    "_pack_topk_routes_post_prefix_kernel": "route-pack-post-prefix",
    "_pack_topk_routes_sort_kernel": "route-pack-sort",
}

_TP_SOURCE_FILES = (Path("sparkinfer/integration/tp_moe.py"),)
_TP_TINY_SOURCE_FILES = (
    Path("sparkinfer/integration/tp_moe.py"),
    Path("sparkinfer/moe/fused/tiny_decode.py"),
)
_W4_SOURCE_FILES = (Path("sparkinfer/moe/fused/w4a16/kernel.py"),)
_W4_ROUTE_SOURCE_FILES = (Path("sparkinfer/moe/fused/w4a16/route_pack.py"),)


@dataclass(frozen=True)
class CaseSpec:
    paired: CaseContract
    artifact_roles: tuple[str, ...]
    live_tensor_names: tuple[str, ...]
    quantization_gate: str

    @property
    def case_id(self) -> str:
        return f"{FAMILY}/{self.paired.name}"

    @property
    def spec_hashes(self) -> tuple[str, ...]:
        return self.paired.spec_hashes

    @property
    def role_by_spec_hash(self) -> dict[str, str]:
        return dict(zip(self.spec_hashes, self.artifact_roles, strict=True))

    @property
    def correctness_gates(self) -> tuple[str, ...]:
        oracle_gate = (
            "bit-exact-independent-gpu-oracle"
            if self.paired.exact_oracle
            else "reviewed-quantized-tolerance"
        )
        return tuple(
            sorted(
                {
                    oracle_gate,
                    "finite",
                    "guard-canaries",
                    "live-input-response",
                    "nonzero",
                    "poison-overwrite",
                    self.quantization_gate,
                }
            )
        )

    @property
    def cross_arm_output_policy(self) -> str:
        return "bit-exact" if self.paired.exact_oracle else "oracle-only"

    @property
    def input_contract(self) -> dict[str, object]:
        return {
            "schema": INPUT_SCHEMA,
            "case_id": self.case_id,
            "paired_case": self.paired.name,
            "corpus_nodeid": self.paired.corpus_nodeid,
            "shape": dict(self.paired.shape),
            "exception_semantic_keys": list(self.paired.exception_semantic_keys),
            "source": {
                "builder": (
                    "validation.cutlass_migration.diagnostics.paired."
                    "compute_exceptions._build_state"
                ),
                "scenario_count": 2,
                **_SOURCE_DESCRIPTIONS[self.paired.name],
            },
            "oracle": {
                "implementation": (
                    "validation.cutlass_migration.diagnostics.paired."
                    "compute_exceptions._correctness"
                ),
                "exact": self.paired.exact_oracle,
                "minimum_cosine": self.paired.min_cosine,
                "maximum_normalized_rmse": self.paired.max_normalized_rmse,
            },
            "controlled_dispatch_environment": {
                "SPARKINFER_W4A8_TINY_DECODE": "1",
                "SPARKINFER_W4A16_SMALL_M_DIRECT": "1",
                "SPARKINFER_MICRO_DYNAMIC_CUTOVER_PAIRS": None,
                "SPARKINFER_DYNAMIC_TILE_MN": None,
            },
        }


CASES = tuple(
    CaseSpec(
        paired=paired,
        artifact_roles=_ARTIFACT_ROLES[name],
        live_tensor_names=_LIVE_TENSOR_NAMES[name],
        quantization_gate=_QUANTIZATION_GATES[name],
    )
    for name, paired in PAIRED_CASES.items()
)
if set(PAIRED_CASES) != set(_ARTIFACT_ROLES) or any(
    len(spec.artifact_roles) != len(spec.paired.object_specs) for spec in CASES
):
    raise RuntimeError("compute-exception single-arm case/object map is incomplete")

CORRECTNESS_GATES = tuple(
    sorted({gate for spec in CASES for gate in spec.correctness_gates})
)


@dataclass(frozen=True)
class GuardedOutput:
    storage: torch.Tensor
    payload_span_elements: int


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_single_arm_arguments(parser)
    parser.add_argument(
        "--regular-replays-per-reported-sample",
        type=int,
        default=1,
        help="event-bracketed replay aggregation for regular exception rows",
    )
    parser.add_argument(
        "--tiny-replays-per-reported-sample",
        type=int,
        default=64,
        help="event-bracketed replay aggregation for timer-quantized W4 rows",
    )
    return parser.parse_args()


def _source_file_records(
    repo_root: Path, paths: Sequence[Path]
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for relative in paths:
        path = (repo_root / relative).resolve()
        try:
            normalized = path.relative_to(repo_root).as_posix()
        except ValueError as exc:
            raise RuntimeError(f"source-owned path escapes repo: {path}") from exc
        if not path.is_file():
            raise RuntimeError(f"source-owned kernel path does not exist: {path}")
        records.append({"path": normalized, "sha256": sha256_file(path)})
    return sorted(records, key=lambda record: record["path"])


def _output_storage_span(output: torch.Tensor) -> int:
    if output.numel() == 0:
        raise RuntimeError("compute-exception output may not be empty")
    if any(stride < 0 for stride in output.stride()):
        raise RuntimeError("compute-exception output has a negative stride")
    return 1 + sum(
        (size - 1) * stride
        for size, stride in zip(output.shape, output.stride(), strict=True)
    )


def _install_output_guards(state: CaseState) -> GuardedOutput:
    output = state.output
    payload_span = _output_storage_span(output)
    storage = torch.full(
        (GUARD_ELEMENTS + payload_span + GUARD_ELEMENTS,),
        GUARD_VALUE,
        dtype=output.dtype,
        device=output.device,
    )
    payload = storage[GUARD_ELEMENTS : GUARD_ELEMENTS + payload_span]
    output.set_(
        payload.untyped_storage(),
        payload.storage_offset(),
        output.shape,
        output.stride(),
    )
    return GuardedOutput(storage=storage, payload_span_elements=payload_span)


def _assert_output_guards(guarded: GuardedOutput) -> None:
    suffix_begin = GUARD_ELEMENTS + guarded.payload_span_elements
    prefix_ok = bool(torch.all(guarded.storage[:GUARD_ELEMENTS] == GUARD_VALUE).item())
    suffix_ok = bool(torch.all(guarded.storage[suffix_begin:] == GUARD_VALUE).item())
    if not prefix_ok or not suffix_ok:
        raise AssertionError("compute-exception output guard canary changed")


def _tensor_binding(tensor: torch.Tensor) -> dict[str, object]:
    return {
        "address": tensor.data_ptr(),
        "storage_address": tensor.untyped_storage().data_ptr(),
        "storage_offset": tensor.storage_offset(),
        "storage_capacity_bytes": tensor.untyped_storage().nbytes(),
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
    }


def _pointer_snapshot(
    state: CaseState, guarded: GuardedOutput
) -> dict[str, dict[str, object]]:
    snapshot = {
        name: _tensor_binding(tensor)
        for name, tensor in sorted(state.stable_tensors.items())
    }
    snapshot["output_guard_storage"] = _tensor_binding(guarded.storage)
    return snapshot


def _tensor_hashes(tensors: Mapping[str, torch.Tensor]) -> dict[str, str]:
    return {name: tensor_sha256(tensor) for name, tensor in sorted(tensors.items())}


def _live_tensors(spec: CaseSpec, state: CaseState) -> dict[str, torch.Tensor]:
    missing = sorted(set(spec.live_tensor_names) - set(state.stable_tensors))
    if missing:
        raise RuntimeError(f"{spec.case_id}: missing live tensors {missing}")
    return {name: state.stable_tensors[name] for name in spec.live_tensor_names}


def _mutable_storage_addresses(spec: CaseSpec, state: CaseState) -> set[int]:
    mutable: set[int] = set()
    for name, tensor in state.stable_tensors.items():
        if (
            name in spec.live_tensor_names
            or name == "output"
            or name.startswith("scratch.")
            or name.startswith("buffers.")
            or name in {"c_tmp", "locks", "intermediate_cache2_micro"}
        ):
            mutable.add(tensor.untyped_storage().data_ptr())
    return mutable


def _owner_tensor_map(spec: CaseSpec, state: CaseState) -> dict[str, torch.Tensor]:
    """Collect immutable tensor leaves, including prepared weights in owners."""

    mutable_storage = _mutable_storage_addresses(spec, state)
    observed_objects: set[int] = set()
    tensors: dict[str, torch.Tensor] = {}

    def visit(value: object, path: str, depth: int) -> None:
        if depth > 12:
            return
        if isinstance(value, torch.Tensor):
            if value.untyped_storage().data_ptr() not in mutable_storage:
                tensors.setdefault(path, value)
            return
        if value is None or isinstance(value, (str, bytes, int, float, bool)):
            return
        identity = id(value)
        if identity in observed_objects:
            return
        observed_objects.add(identity)
        if isinstance(value, Mapping):
            for key in sorted(value, key=lambda item: str(item)):
                visit(value[key], f"{path}.{key}", depth + 1)
            return
        if isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                visit(item, f"{path}.{index}", depth + 1)
            return
        if is_dataclass(value) and not isinstance(value, type):
            for field in fields(value):
                visit(getattr(value, field.name), f"{path}.{field.name}", depth + 1)
            return
        module = type(value).__module__
        if module.startswith(
            ("sparkinfer.", "tests.", "benchmarks.", "validation.cutlass_migration.")
        ) and hasattr(value, "__dict__"):
            for name, item in sorted(vars(value).items()):
                if not name.startswith("_"):
                    visit(item, f"{path}.{name}", depth + 1)

    for name, tensor in sorted(state.read_only_tensors.items()):
        visit(tensor, f"declared.{name}", 0)
    visit(state.owners, "owners", 0)
    if not tensors:
        raise RuntimeError(f"{spec.case_id}: immutable tensor set is empty")
    return tensors


def _workspace_capacity_bytes(state: CaseState, guarded: GuardedOutput) -> int:
    storages: dict[int, int] = {}
    for tensor in (*state.stable_tensors.values(), guarded.storage):
        address = tensor.untyped_storage().data_ptr()
        storages[address] = tensor.untyped_storage().nbytes()
    return sum(storages.values())


def _load_case_artifacts(
    *,
    spec: CaseSpec,
    cache: Path,
    runtime_fingerprint: str,
) -> tuple[
    dict[str, object],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    compiled: dict[str, object] = {}
    provenance: dict[str, dict[str, Any]] = {}
    artifact_before: dict[str, dict[str, Any]] = {}
    observed_comparison_keys: set[str] = set()
    for kernel_id, spec_hash, target in spec.paired.object_specs:
        exact, record = load_exact(cache.resolve(), spec_hash)
        if record.get("kernel_id") != kernel_id:
            raise RuntimeError(f"{spec.case_id}/{spec_hash}: kernel ID differs")
        if record.get("package_fingerprint") != runtime_fingerprint:
            raise RuntimeError(
                f"{spec.case_id}/{spec_hash}: object/runtime fingerprints differ"
            )
        manifest_path = Path(str(record["manifest_path"]))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("target") != target:
            raise RuntimeError(f"{spec.case_id}/{spec_hash}: target differs")
        launch_metadata = manifest.get("launch_metadata")
        if (
            not isinstance(launch_metadata, Mapping)
            or launch_metadata.get("status") != "exact"
            or not isinstance(launch_metadata.get("launch_dynamic_smem_bytes"), Mapping)
        ):
            raise RuntimeError(
                f"{spec.case_id}/{spec_hash}: launch metadata is not exact"
            )
        comparison_key = comparison_semantic_key_from_manifest(manifest)
        observed_comparison_keys.add(comparison_key)
        enriched = {
            **record,
            "target": target,
            "launch_metadata": launch_metadata,
            "comparison_semantic_key": comparison_key,
        }
        before = verify_artifact(enriched)
        # Fail before graph construction if the PTX sidecar or any raw-chain
        # evidence is absent.  The final evidence is rebuilt with after==before.
        exact_artifact_evidence(
            enriched,
            verification_before=before,
            verification_after=before,
        )
        compiled[spec_hash] = exact
        provenance[spec_hash] = enriched
        artifact_before[spec_hash] = before
    expected_comparison_keys = set(spec.paired.exception_semantic_keys)
    if not expected_comparison_keys <= observed_comparison_keys:
        raise RuntimeError(
            f"{spec.case_id}: exact objects do not cover exception semantic keys"
        )
    return compiled, provenance, artifact_before


def _kernel_metadata(node: Mapping[str, Any]) -> tuple[str, list[int], list[int], int]:
    kernel_name = str(node.get("kernel_name", ""))
    grid = node.get("grid")
    block = node.get("block")
    dynamic_smem = node.get("dynamic_smem_bytes")
    if (
        not kernel_name
        or not isinstance(grid, list)
        or len(grid) != 3
        or not all(isinstance(value, int) for value in grid)
        or not isinstance(block, list)
        or len(block) != 3
        or not all(isinstance(value, int) for value in block)
        or not isinstance(dynamic_smem, int)
        or isinstance(dynamic_smem, bool)
        or dynamic_smem < 0
    ):
        raise RuntimeError(f"malformed CUDA graph kernel metadata: {node}")
    return kernel_name, list(grid), list(block), dynamic_smem


def _source_owned_node(
    *,
    spec: CaseSpec,
    repo_root: Path,
    node_index: int,
    kernel_name: str,
    grid: list[int],
    block: list[int],
    dynamic_smem_bytes: int,
) -> dict[str, object]:
    route_matches = [
        (symbol, role)
        for symbol, role in _ROUTE_PACK_ROLES.items()
        if symbol == kernel_name or symbol in kernel_name
    ]
    if len(route_matches) == 1 and spec.paired.name == "w4a16-swiglu-limit-m24":
        _, role = route_matches[0]
        implementation = "triton"
        source_paths = _W4_ROUTE_SOURCE_FILES
    else:
        lowered = kernel_name.lower()
        is_fill = "fillfunctor" in lowered or "fill_kernel" in lowered
        is_copy = "copy_kernel" in lowered or "direct_copy" in lowered
        torch_source_paths: tuple[Path, ...] | None = None
        if spec.paired.name == "tp-moe-nvfp4-micro-m2":
            torch_source_paths = _TP_SOURCE_FILES
        elif spec.paired.name == "tp-moe-w4a8-mx-tiny-m2":
            torch_source_paths = _TP_TINY_SOURCE_FILES
        elif spec.paired.name == "w4a16-native-e8m0-small-m1":
            torch_source_paths = _W4_SOURCE_FILES
        if torch_source_paths is None or not (is_fill or is_copy):
            raise RuntimeError(
                f"{spec.case_id}: unclassified non-CUTLASS graph kernel "
                f"ordinal={node_index}, name={kernel_name!r}"
            )
        operation = "fill" if is_fill else "copy"
        role = f"torch-{operation}-{node_index}"
        implementation = "torch_cuda"
        source_paths = torch_source_paths
    return {
        "node_index": node_index,
        "role": role,
        "implementation": implementation,
        "kernel_name": kernel_name,
        "kernel_name_sha256": hashlib.sha256(kernel_name.encode("utf-8")).hexdigest(),
        "grid": grid,
        "block": block,
        "dynamic_smem_bytes": dynamic_smem_bytes,
        "source_files": _source_file_records(repo_root, source_paths),
    }


def _classify_graph_kernel_nodes(
    *,
    spec: CaseSpec,
    topology: Mapping[str, Any],
    provenance: Mapping[str, Mapping[str, Any]],
    repo_root: Path,
) -> tuple[tuple[tuple[int, str], ...], list[dict[str, object]]]:
    raw_nodes = topology.get("nodes")
    if not isinstance(raw_nodes, list):
        raise RuntimeError(f"{spec.case_id}: graph topology has no node list")
    kernel_nodes = [
        node
        for node in raw_nodes
        if isinstance(node, Mapping) and node.get("type") == "CU_GRAPH_NODE_TYPE_KERNEL"
    ]
    role_by_spec = spec.role_by_spec_hash
    exact_name_roles: dict[str, set[str]] = {}
    exact_name_smem: dict[tuple[str, str], set[int]] = {}
    for spec_hash, record in provenance.items():
        role = role_by_spec[spec_hash]
        launch_smem = record["launch_metadata"]["launch_dynamic_smem_bytes"]
        for name, values in launch_smem.items():
            exact_name_roles.setdefault(str(name), set()).add(role)
            exact_name_smem[(str(name), role)] = {int(value) for value in values}

    exact_nodes: list[tuple[int, str]] = []
    source_owned: list[dict[str, object]] = []
    for kernel_ordinal, node in enumerate(kernel_nodes):
        kernel_name, grid, block, dynamic_smem = _kernel_metadata(node)
        matching_roles = exact_name_roles.get(kernel_name, set())
        if matching_roles:
            if len(matching_roles) != 1:
                raise RuntimeError(
                    f"{spec.case_id}: exact kernel name is ambiguous: {kernel_name}"
                )
            role = next(iter(matching_roles))
            if dynamic_smem not in exact_name_smem[(kernel_name, role)]:
                raise RuntimeError(
                    f"{spec.case_id}: exact kernel SMEM differs for {kernel_name}"
                )
            exact_nodes.append((kernel_ordinal, role))
            continue
        source_owned.append(
            _source_owned_node(
                spec=spec,
                repo_root=repo_root,
                node_index=kernel_ordinal,
                kernel_name=kernel_name,
                grid=grid,
                block=block,
                dynamic_smem_bytes=dynamic_smem,
            )
        )

    observed_roles = {role for _, role in exact_nodes}
    if observed_roles != set(spec.artifact_roles):
        raise RuntimeError(
            f"{spec.case_id}: exact graph role coverage differs: "
            f"observed={sorted(observed_roles)}, expected={sorted(spec.artifact_roles)}"
        )
    covered_indices = sorted(
        [index for index, _ in exact_nodes]
        + [int(record["node_index"]) for record in source_owned]
    )
    if covered_indices != list(range(len(kernel_nodes))):
        raise RuntimeError(f"{spec.case_id}: graph kernel partition is incomplete")
    return tuple(exact_nodes), source_owned


def _replay_checked(
    *,
    spec: CaseSpec,
    state: CaseState,
    graph: torch.cuda.CUDAGraph,
    stream: torch.cuda.Stream,
    guarded: GuardedOutput,
    scenario: int,
    poison: float,
) -> tuple[dict[str, object], torch.Tensor]:
    with torch.cuda.stream(stream):
        state.output.fill_(poison)
    stream.synchronize()
    before = allocator_counters()
    with torch.cuda.stream(stream):
        graph.replay()
    stream.synchronize()
    after = allocator_counters()
    if after != before:
        raise AssertionError(
            f"{spec.case_id}: correctness replay allocated: {before}->{after}"
        )
    if bool(torch.isnan(state.output.float()).any().item()):
        raise AssertionError(f"{spec.case_id}: graph left poisoned output values")
    metrics = _correctness(spec.paired, state.output, state.references[scenario])
    _assert_output_guards(guarded)
    return metrics, state.output.clone()


def _run_case(
    *,
    spec: CaseSpec,
    reviewed: Mapping[str, Any],
    arm: str,
    repo_root: Path,
    compiled: Mapping[str, object],
    provenance: Mapping[str, Mapping[str, Any]],
    artifact_before: Mapping[str, Mapping[str, Any]],
    precondition: int,
    precondition_seconds: float,
    maximum_precondition_seconds: float,
    warmup: int,
    replays: int,
    event_batch_replays: int,
    expected_physical_gpu: int,
    max_sm_clock_delta_mhz: float,
    l2_flush_bytes: int,
    replays_per_reported_sample: int,
) -> dict[str, object]:
    for spec_hash, role in spec.role_by_spec_hash.items():
        verify_case_compile_contract(
            case_id=spec.case_id,
            reviewed=reviewed,
            arm=arm,
            role=role,
            provenance=provenance[spec_hash],
        )

    state = _build_state(spec.paired)
    guarded = _install_output_guards(state)
    torch.cuda.synchronize()
    _assert_output_guards(guarded)
    if torch.equal(state.references[0], state.references[1]):
        raise AssertionError(f"{spec.case_id}: oracle scenarios are not distinct")

    state.install(0)
    torch.cuda.synchronize()
    live = _live_tensors(spec, state)
    baseline_live_hashes = _tensor_hashes(live)
    immutable = _owner_tensor_map(spec, state)
    immutable_hashes_before = _tensor_hashes(immutable)
    pointers_before = _pointer_snapshot(state, guarded)
    workspace_capacity_bytes = _workspace_capacity_bytes(state, guarded)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    state.clear_resolution_cache()
    observed_specs: list[str] = []
    compile_misses_before = int(cute_compiler.compile_cache_info()["compile_misses"])
    with _exact_compile_resolution(state.resolver_module, compiled, observed_specs):
        with torch.cuda.stream(stream):
            state.output.fill_(math.nan)
            eager_result = state.launch()
        stream.synchronize()
        if eager_result.data_ptr() != state.output.data_ptr():
            raise AssertionError(f"{spec.case_id}: production route replaced output")
        _correctness(spec.paired, state.output, state.references[0])
        _assert_output_guards(guarded)
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph, stream=stream):
            state.launch()
        stream.synchronize()
    compile_misses_after = int(cute_compiler.compile_cache_info()["compile_misses"])
    if compile_misses_after != compile_misses_before:
        raise AssertionError(f"{spec.case_id}: graph capture compiled a CUTLASS kernel")
    state.clear_resolution_cache()
    if set(observed_specs) != set(spec.spec_hashes):
        raise RuntimeError(
            f"{spec.case_id}: production route resolved {observed_specs}, "
            f"expected={list(spec.spec_hashes)}"
        )

    full_topology = graph_topology(graph)
    _validate_manifest_kernels(full_topology, provenance)
    topology = single_graph_topology(graph)
    if (
        reviewed.get("_discovery") is not True
        and topology != reviewed["graph_topology_contract"][arm]
    ):
        raise RuntimeError(f"{spec.case_id}: graph topology differs from review")
    exact_nodes, source_owned_nodes = _classify_graph_kernel_nodes(
        spec=spec,
        topology=full_topology,
        provenance=provenance,
        repo_root=repo_root,
    )
    reviewed_source_owned = reviewed["compile_artifact_contract"][arm].get(
        "source_owned_kernel_nodes"
    )
    if (
        reviewed.get("_discovery") is not True
        and source_owned_nodes != reviewed_source_owned
    ):
        raise RuntimeError(
            f"{spec.case_id}: source-owned graph nodes differ from review"
        )

    baseline_metrics: dict[str, object] | None = None
    baseline_output: torch.Tensor | None = None
    for poison in (float("nan"), -321.0):
        baseline_metrics, baseline_output = _replay_checked(
            spec=spec,
            state=state,
            graph=graph,
            stream=stream,
            guarded=guarded,
            scenario=0,
            poison=poison,
        )
    if baseline_metrics is None or baseline_output is None:
        raise AssertionError(f"{spec.case_id}: baseline replay did not run")

    state.install(1)
    torch.cuda.synchronize()
    mutated_live_hashes = _tensor_hashes(live)
    if mutated_live_hashes == baseline_live_hashes:
        raise AssertionError(f"{spec.case_id}: live-input mutation was ineffective")
    mutated_metrics: dict[str, object] | None = None
    mutated_output: torch.Tensor | None = None
    for poison in (float("nan"), 321.0):
        mutated_metrics, mutated_output = _replay_checked(
            spec=spec,
            state=state,
            graph=graph,
            stream=stream,
            guarded=guarded,
            scenario=1,
            poison=poison,
        )
    if (
        mutated_metrics is None
        or mutated_output is None
        or torch.equal(baseline_output, mutated_output)
    ):
        raise AssertionError(f"{spec.case_id}: live input did not change output")

    state.install(0)
    torch.cuda.synchronize()
    if _tensor_hashes(live) != baseline_live_hashes:
        raise AssertionError(f"{spec.case_id}: live inputs did not restore exactly")

    conditions, allocation_records = time_single_graph_conditions(
        graph,
        precondition=precondition,
        warmup=warmup,
        replays=replays,
        stream=stream,
        l2_flush_bytes=l2_flush_bytes,
        replays_per_reported_sample=replays_per_reported_sample,
        event_batch_replays=event_batch_replays,
        precondition_seconds=precondition_seconds,
        maximum_precondition_seconds=maximum_precondition_seconds,
        mode_snapshot=lambda: gpu_mode_snapshot(expected_physical_gpu),
        required_pstate="P1",
        max_sm_clock_delta_mhz=max_sm_clock_delta_mhz,
    )
    post_metrics, _ = _replay_checked(
        spec=spec,
        state=state,
        graph=graph,
        stream=stream,
        guarded=guarded,
        scenario=0,
        poison=float("nan"),
    )
    if _tensor_hashes(live) != baseline_live_hashes:
        raise AssertionError(f"{spec.case_id}: timing changed live inputs")
    if _pointer_snapshot(state, guarded) != pointers_before:
        raise AssertionError(f"{spec.case_id}: tensor addresses/capacities changed")
    if single_graph_topology(graph) != topology:
        raise AssertionError(f"{spec.case_id}: graph topology changed")
    if _tensor_hashes(immutable) != immutable_hashes_before:
        raise AssertionError(f"{spec.case_id}: immutable tensors changed")
    _assert_output_guards(guarded)

    artifacts: list[dict[str, object]] = []
    for spec_hash, role in spec.role_by_spec_hash.items():
        after = verify_artifact(provenance[spec_hash])
        evidence = exact_artifact_evidence(
            provenance[spec_hash],
            verification_before=artifact_before[spec_hash],
            verification_after=after,
        )
        artifacts.append(bind_exact_artifact(role=role, evidence=evidence))
    launch_plan = build_exact_launch_plan(
        case_id=spec.case_id,
        reviewed=reviewed,
        arm=arm,
        artifacts=artifacts,
        observed_roles=exact_nodes,
    )
    covered_indices = sorted(
        [int(binding["node_index"]) for binding in launch_plan]
        + [int(binding["node_index"]) for binding in source_owned_nodes]
    )
    if covered_indices != list(range(int(topology["kernel_node_count"]))):
        raise AssertionError(f"{spec.case_id}: graph kernel coverage is incomplete")

    if not bool(post_metrics["passed"]):
        raise AssertionError(f"{spec.case_id}: post-timing oracle failed")
    output_identity = (
        str(baseline_metrics["actual_sha256"])
        if spec.paired.exact_oracle
        else tensor_sha256(state.references[0])
    )
    read_only_inputs_sha256 = json_sha256(
        {
            "immutable_tensors": immutable_hashes_before,
            "baseline_live_inputs": baseline_live_hashes,
            "scenario_references": [
                tensor_sha256(reference) for reference in state.references
            ],
        }
    )
    allocation = allocation_records["warm_l2"]
    return {
        "case_id": spec.case_id,
        "case_contract_sha256": reviewed["case_contract_sha256"],
        "input_sha256": json_sha256(spec.input_contract),
        "artifacts": artifacts,
        "launch_plan": launch_plan,
        "source_owned_kernel_nodes": source_owned_nodes,
        "correctness": {
            "independent_oracle": True,
            "oracle": "compute-exception-independent-gpu-quantized-reference",
            "passed": True,
            "finite": bool(baseline_metrics["finite"]),
            "nonzero_count": int(baseline_metrics["nonzero"]),
            "gates": {gate: True for gate in spec.correctness_gates},
            "read_only_inputs_immutable": True,
            "read_only_inputs_sha256": read_only_inputs_sha256,
            "output_sha256": output_identity,
        },
        "graph": {
            "capture_passed": True,
            "replay_passed": True,
            "topology_stable": True,
            "addresses_stable": True,
            "live_input_changed_output": True,
            "poison_overwrite_passed": True,
            **topology,
        },
        "allocation": {
            "fixed_workspace_capacity": True,
            "workspace_capacity_bytes": workspace_capacity_bytes,
            "stable_addresses": True,
            "allocator_stable": True,
            "zero_replay_allocations": True,
            **allocation,
            "condition_counters": allocation_records,
        },
        "conditions": conditions,
    }


def main() -> int:
    args = _args()
    if args.regular_replays_per_reported_sample < 1:
        raise ValueError("regular replay aggregation must be positive")
    if args.tiny_replays_per_reported_sample < 1:
        raise ValueError("tiny replay aggregation must be positive")

    producer_path = Path(__file__).resolve()
    session = begin_single_arm_session(
        args,
        family=FAMILY,
        producer_path=producer_path,
        bindings=tuple(
            ReviewedCaseBinding(
                case_id=spec.case_id,
                input_sha256=json_sha256(spec.input_contract),
                correctness_gates=spec.correctness_gates,
                cross_arm_output_policy=spec.cross_arm_output_policy,
            )
            for spec in CASES
        ),
    )

    loaded: dict[
        str,
        tuple[
            dict[str, object],
            dict[str, dict[str, Any]],
            dict[str, dict[str, Any]],
        ],
    ] = {}
    for spec in CASES:
        loaded[spec.paired.name] = _load_case_artifacts(
            spec=spec,
            cache=args.cache,
            runtime_fingerprint=session.runtime_fingerprint,
        )

    with _controlled_dispatch_environment():
        cases = []
        for spec in CASES:
            compiled, provenance, artifact_before = loaded[spec.paired.name]
            cases.append(
                _run_case(
                    spec=spec,
                    reviewed=session.reviewed_cases[spec.case_id],
                    arm=session.arm,
                    repo_root=session.repo_root,
                    compiled=compiled,
                    provenance=provenance,
                    artifact_before=artifact_before,
                    precondition=args.precondition,
                    precondition_seconds=args.precondition_seconds,
                    maximum_precondition_seconds=args.maximum_precondition_seconds,
                    warmup=args.warmup,
                    replays=args.replays,
                    event_batch_replays=args.event_batch_replays,
                    expected_physical_gpu=session.expected_physical_gpu,
                    max_sm_clock_delta_mhz=args.max_sm_clock_delta_mhz,
                    l2_flush_bytes=args.l2_flush_bytes,
                    replays_per_reported_sample=(
                        args.tiny_replays_per_reported_sample
                        if spec.paired.name in _TIMER_LIMITED_CASES
                        else args.regular_replays_per_reported_sample
                    ),
                )
            )
    finish_single_arm_session(session, cases)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
