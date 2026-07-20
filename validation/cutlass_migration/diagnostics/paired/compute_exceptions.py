#!/usr/bin/env python3
"""Exact-object CUDA-graph ABBA for the compute exception rows.

This benchmark covers the final-v5 non-attention exception rows that are not
already exercised by a family-specific ABBA adapter:

* dense NVFP4, fused-quant MXFP8, and grouped fused-quant MXFP8 GEMM;
* TP-MoE NVFP4 micro decode and two-phase W4A8-MX tiny decode;
* standalone W4A16 GEMM, native-E8M0 W4A16 small-M decode, and the
  W4A16 SwiGLU-limit fused specialization.

Every arm is resolved from an immutable compile-cache object.  The production
launcher is allowed to construct its normal tensor wrapper, but its module-local
``sparkinfer_compile`` is replaced by an exact-spec resolver; any unlisted compile
request aborts.  Both arms capture against the same input, output, and workspace
addresses.  GPU oracle checks, live-input mutation, full-output overwrite,
zero-allocation replay, graph topology, warm/cold-L2 ABBA, artifact hashes, and
GPU mode are recorded before an output report is accepted.

This is GPU-only benchmark evidence, not a CPU acceptance path.  It requires
physical GPU 4 or 5 and SM120.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType
from typing import Any

import torch

from benchmarks.common import nvidia_smi_gpu_mode_snapshot
from validation.cutlass_migration.core.evidence_status import (
    add_evidence_status_argument,
)
from validation.cutlass_migration.core.comparison_identity import (
    comparison_semantic_key_from_manifest,
)
from validation.cutlass_migration.core.exact_cache_abba import (
    allocator_counters,
    graph_topology,
    load_exact,
    sha256_file,
    tensor_sha256,
    time_conditions,
    topology_signature,
    verify_artifact,
)
from validation.cutlass_migration.paths import REPO_ROOT
import sparkinfer.cute.compiler as cute_compiler


@dataclass(frozen=True)
class CaseContract:
    name: str
    exception_semantic_keys: tuple[str, ...]
    object_specs: tuple[tuple[str, str, str], ...]
    corpus_nodeid: str
    shape: Mapping[str, object]
    exact_oracle: bool
    min_cosine: float = 1.0
    max_normalized_rmse: float = 0.0

    @property
    def spec_hashes(self) -> tuple[str, ...]:
        return tuple(spec_hash for _, spec_hash, _ in self.object_specs)


_CASES = {
    case.name: case
    for case in (
        CaseContract(
            name="dense-nvfp4-m32",
            exception_semantic_keys=(
                "14c6d8c0a8dd7d1c99b2de52522e021a9ceaed6aef3d3007c4d2ed1bdc62b21f",
            ),
            object_specs=(
                (
                    "gemm.dense",
                    "23b007fedc8c48cfbe86fabc93996c305e3142a66ac23730c216e316588857ba",
                    "sparkinfer.gemm.dense._DenseGemmLaunch",
                ),
            ),
            corpus_nodeid=(
                "tests/test_cute_migration_gemm_corpus.py::"
                "test_cute_migration_dense_nvfp4_gpu_oracle_and_graph"
            ),
            shape={"m": 32, "n": 128, "k": 128, "groups": 1},
            exact_oracle=True,
        ),
        CaseContract(
            name="dense-fused-quant-m2",
            exception_semantic_keys=(
                "20b103dbc55c2df512072fc80a64b8cfc695b19c7faa3dced09a93411c051aae",
            ),
            object_specs=(
                (
                    "gemm.dense_fused_quant_a",
                    "1e11604b95ff4b527fa84ec9c8615ff62bb36169ee5956e6671848148d54ea13",
                    "sparkinfer.gemm.dense._DenseGemmFusedQuantALaunch",
                ),
            ),
            corpus_nodeid=(
                "tests/test_cute_migration_gemm_corpus.py::"
                "test_cute_migration_dense_fused_quant_gpu_oracle_and_graph"
            ),
            shape={"m": 2, "n": 128, "k": 128, "groups": 1},
            exact_oracle=True,
        ),
        CaseContract(
            name="dense-grouped-fused-quant-m2-g2",
            exception_semantic_keys=(
                "d1e3cb157c417a90c9cad6f4a5f566ec6ebc359f763598fd371c2f33399bf44f",
            ),
            object_specs=(
                (
                    "gemm.dense_fused_quant_a_grouped",
                    "55325c809bb30e4f47abf0bddff6a43a25f202c1dde4cd6738bb34ca4077560e",
                    "sparkinfer.gemm.dense._DenseGemmFusedQuantAGroupedLaunch",
                ),
            ),
            corpus_nodeid=(
                "tests/test_cute_migration_gemm_corpus.py::"
                "test_cute_migration_dense_grouped_fused_quant_gpu_oracle_and_graph"
            ),
            shape={"m": 2, "n": 128, "k": 128, "groups": 2},
            exact_oracle=True,
        ),
        CaseContract(
            name="tp-moe-nvfp4-micro-m2",
            exception_semantic_keys=(
                "9e1f4a3add39c2af646d7b4b813982295083a0e62de2eefe83b0cca0b4c41552",
            ),
            object_specs=(
                (
                    "integration.tp_moe.micro_direct",
                    "9410c16b77c3f5eaa4a4a3b2635de5cb2021ca0d69cd9a572d7a0a1eacb3b071",
                    "sparkinfer.moe.fused.silu.MoEMicroKernelSilu",
                ),
            ),
            corpus_nodeid=(
                "tests/test_cute_migration_moe_standard_corpus.py::"
                "test_standard_moe_micro_live_graph_oracle"
            ),
            shape={
                "m": 2,
                "experts": 4,
                "hidden": 512,
                "intermediate": 128,
                "topk": 2,
                "quant_mode": "nvfp4",
            },
            exact_oracle=False,
            min_cosine=0.999,
            max_normalized_rmse=0.03,
        ),
        CaseContract(
            name="tp-moe-w4a8-mx-tiny-m2",
            exception_semantic_keys=(
                "284e82282cf14ee2d6cde4cb0968b510b219aad9b08e4e6e2fd9b1446f5e2eca",
            ),
            object_specs=(
                (
                    "integration.tp_moe.tiny_decode",
                    "d62452570aa283a5fceda983147c4cfd2d8240f20d60e6886f61b644e1d735b7",
                    "sparkinfer.moe.fused.tiny_decode.MoETinyDecodeKernelBackendPhase1",
                ),
                (
                    "integration.tp_moe.tiny_decode",
                    "a907a087fca9ba405e4a11fe532d6b30d17088ddb6a599e58df6e049ee152738",
                    "sparkinfer.moe.fused.tiny_decode.MoETinyDecodeKernelBackendPhase2",
                ),
            ),
            corpus_nodeid=(
                "tests/test_cute_migration_moe_standard_corpus.py::"
                "test_standard_moe_tiny_decode_live_graph_oracle"
            ),
            shape={
                "m": 2,
                "experts": 4,
                "hidden": 512,
                "intermediate": 128,
                "topk": 2,
                "quant_mode": "w4a8_mx",
            },
            exact_oracle=False,
            min_cosine=0.998,
            max_normalized_rmse=0.05,
        ),
        CaseContract(
            name="w4a16-standalone-gemm-m128",
            exception_semantic_keys=(
                "db9920b7de8b664b4a3ff9c8abde88df4758ba32432ee7998f94ca083370f9b9",
            ),
            object_specs=(
                (
                    "moe.w4a16.gemm",
                    "b8786f12e68d95b6436c8619812a07db2cb6f6f39dd727c85cee3415cd5df81c",
                    "sparkinfer.moe.fused.w4a16.kernel.W4A16GemmKernel",
                ),
            ),
            corpus_nodeid=(
                "tests/test_w4a16_standalone_kernels.py::"
                "test_w4a16_standalone_gemm_and_activation_match_gpu_oracles_under_graph"
            ),
            shape={
                "m": 128,
                "experts": 8,
                "hidden": 128,
                "intermediate": 128,
                "topk": 4,
                "activation": "silu",
            },
            exact_oracle=False,
            min_cosine=0.9975,
            max_normalized_rmse=0.15,
        ),
        CaseContract(
            name="w4a16-native-e8m0-small-m1",
            exception_semantic_keys=(
                "d2791ff09c2abcc3d7f0b18636b1e5878ce4bf2e57e6ab46c2011cfede1e0066",
            ),
            object_specs=(
                (
                    "moe.w4a16.small_m_direct",
                    "5f2f4d1dc21b086d371315981ac5b5229deefa0bad4ce1dbdfb921faaef6a1d5",
                    ("sparkinfer.moe.fused.w4a16.kernel.MoEMicroKernelW4A16SmallMDirect"),
                ),
            ),
            corpus_nodeid=(
                "tests/test_w4a16_e2e.py::"
                "test_w4a16_e8m0_native_micro_matches_raw_e8m0_oracle"
            ),
            shape={
                "m": 1,
                "experts": 4,
                "hidden": 128,
                "intermediate": 128,
                "topk": 2,
                "activation": "silu",
                "scale_format": "e8m0_k32",
            },
            exact_oracle=False,
            min_cosine=0.9975,
            max_normalized_rmse=0.15,
        ),
        CaseContract(
            name="w4a16-swiglu-limit-m24",
            exception_semantic_keys=(
                "5b8958fcfe41f8b561a83ccd8ebf47cbe307cac8c041619751800605b54d719a",
            ),
            object_specs=(
                (
                    "moe.w4a16.fused_moe",
                    "ccfeba5ceedeac848bf224e8508f28a9ef6a90f1e53d10f9dc8812f38745c415",
                    "sparkinfer.moe.fused.w4a16.kernel.W4A16FusedMoeKernel",
                ),
                (
                    "moe.w4a16.topk_sum",
                    "4622cf1512d6a83ed3233c7fa5254e2544067f673c3b608d197388ba9c06845a",
                    "sparkinfer.moe.fused.w4a16.kernel.W4A16TopKSumKernel",
                ),
            ),
            corpus_nodeid=(
                "tests/test_w4a16_e2e.py::"
                "test_w4a16_moe_swiglu_limit_matches_oracle_under_cuda_graph"
            ),
            shape={
                "m": 24,
                "experts": 8,
                "hidden": 128,
                "intermediate": 128,
                "topk": 2,
                "activation": "silu",
                "swiglu_limit": 10.0,
            },
            exact_oracle=False,
            min_cosine=0.9975,
            max_normalized_rmse=0.15,
        ),
    )
}
_TIMER_LIMITED_CASES = {
    "w4a16-standalone-gemm-m128",
    "w4a16-native-e8m0-small-m1",
    "w4a16-swiglu-limit-m24",
}


@dataclass
class CaseState:
    launch: Callable[[], torch.Tensor]
    install: Callable[[int], None]
    output: torch.Tensor
    references: tuple[torch.Tensor, torch.Tensor]
    resolver_module: ModuleType
    clear_resolution_cache: Callable[[], None]
    stable_tensors: Mapping[str, torch.Tensor]
    read_only_tensors: Mapping[str, torch.Tensor]
    workspace: Mapping[str, object]
    owners: tuple[object, ...]


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_evidence_status_argument(parser)
    parser.add_argument("--a-cache", type=Path, required=False)
    parser.add_argument("--a-label", default="cutlass-4.5.2")
    parser.add_argument("--a-cutlass-version", default="4.5.2")
    parser.add_argument("--b-cache", type=Path, required=False)
    parser.add_argument("--b-label", default="cutlass-4.6.0")
    parser.add_argument("--b-cutlass-version", default="4.6.0")
    parser.add_argument(
        "--case",
        action="append",
        choices=tuple(_CASES),
        default=[],
        help="case to run; repeat as needed (default: all rows)",
    )
    parser.add_argument("--precondition-cycles", type=int, default=250)
    parser.add_argument(
        "--precondition-seconds",
        type=float,
        default=5.0,
        help="minimum balanced target-graph activity per cache condition",
    )
    parser.add_argument(
        "--maximum-precondition-seconds",
        type=float,
        default=30.0,
        help="fail if balanced target work cannot reach P1 within this duration",
    )
    parser.add_argument(
        "--max-sm-clock-delta-mhz",
        type=float,
        default=60.0,
        help="maximum P1 SM-clock change allowed across one timing condition",
    )
    parser.add_argument("--warmup-cycles", type=int, default=100)
    parser.add_argument("--cycles", type=int, default=600)
    parser.add_argument("--event-batch-cycles", type=int, default=25)
    parser.add_argument(
        "--tiny-replays-per-reported-sample",
        type=int,
        default=64,
        help="independently event-bracketed graph replays averaged per tiny sample",
    )
    parser.add_argument("--l2-flush-bytes", type=int, default=0)
    parser.add_argument(
        "--expected-physical-gpu", type=int, choices=(4, 5), required=False
    )
    parser.add_argument(
        "--expected-package-fingerprint",
        help="required final-source package fingerprint for both cache arms and host",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="print the frozen row/spec map without running benchmark acceptance",
    )
    return parser.parse_args()


def _case_manifest() -> dict[str, object]:
    return {
        name: {
            "exception_semantic_keys": list(case.exception_semantic_keys),
            "objects": [
                {"kernel_id": kernel_id, "spec_hash": spec_hash, "target": target}
                for kernel_id, spec_hash, target in case.object_specs
            ],
            "corpus_nodeid": case.corpus_nodeid,
            "shape": dict(case.shape),
        }
        for name, case in _CASES.items()
    }


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _toolchain_version(provenance: Mapping[str, Any], package: str) -> str:
    for item in provenance.get("toolchain", []):
        if isinstance(item, Sequence) and len(item) >= 2 and item[0] == package:
            return str(item[1])
    return "missing"


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _verify_ptx_sidecar(provenance: Mapping[str, Any]) -> dict[str, object]:
    manifest_path = Path(str(provenance["manifest_path"]))
    sidecar = manifest_path.with_suffix(".ptx.json")
    if not sidecar.is_file():
        raise RuntimeError(f"missing exact-object PTX evidence: {sidecar}")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    obj = payload.get("object")
    if (
        payload.get("schema") != "sparkinfer.cute.frontend_ptx.v3"
        or payload.get("cache_key") != provenance["cache_key"]
        or payload.get("compile_spec_hash") != provenance["compile_spec_hash"]
        or not isinstance(obj, Mapping)
        or obj.get("sha256") != provenance["object_sha256"]
        or int(obj.get("bytes", -1)) != int(provenance["object_bytes"])
    ):
        raise RuntimeError(f"PTX evidence does not describe exact object: {sidecar}")
    return {
        "path": str(sidecar),
        "sha256": sha256_file(sidecar),
        "source_ptxas": payload.get("source_ptxas"),
        "embedded_cubin_sha256": obj.get("embedded_cubin_sha256"),
    }


def _load_case_artifacts(
    cache: Path,
    case: CaseContract,
    *,
    expected_cutlass: str,
    expected_fingerprint: str,
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    compiled: dict[str, object] = {}
    provenance: dict[str, dict[str, object]] = {}
    for kernel_id, spec_hash, target in case.object_specs:
        object_value, object_provenance = load_exact(cache, spec_hash)
        if object_provenance["kernel_id"] != kernel_id:
            raise RuntimeError(
                f"{case.name}/{spec_hash}: kernel ID mismatch "
                f"{object_provenance['kernel_id']!r} != {kernel_id!r}"
            )
        manifest = json.loads(
            Path(str(object_provenance["manifest_path"])).read_text(encoding="utf-8")
        )
        if manifest.get("target") != target:
            raise RuntimeError(
                f"{case.name}/{spec_hash}: target mismatch "
                f"{manifest.get('target')!r} != {target!r}"
            )
        if object_provenance["package_fingerprint"] != expected_fingerprint:
            raise RuntimeError(f"{case.name}/{spec_hash}: package fingerprint mismatch")
        observed_cutlass = _toolchain_version(object_provenance, "cutlass_dsl")
        if observed_cutlass != expected_cutlass:
            raise RuntimeError(
                f"{case.name}/{spec_hash}: expected CUTLASS {expected_cutlass}, "
                f"manifest reports {observed_cutlass}"
            )
        launch_metadata = manifest.get("launch_metadata")
        if (
            not isinstance(launch_metadata, Mapping)
            or launch_metadata.get("status") != "exact"
            or not isinstance(launch_metadata.get("launch_dynamic_smem_bytes"), Mapping)
        ):
            raise RuntimeError(f"{case.name}/{spec_hash}: launch metadata is not exact")
        object_provenance = {
            **object_provenance,
            "target": target,
            "semantic_key": manifest.get("semantic_key"),
            "comparison_semantic_key": comparison_semantic_key_from_manifest(manifest),
            "launch_metadata": launch_metadata,
            "ptx_evidence": _verify_ptx_sidecar(object_provenance),
        }
        semantic_key = object_provenance["semantic_key"]
        if not _is_sha256(semantic_key):
            raise RuntimeError(
                f"{case.name}/{spec_hash}: object manifest semantic key is invalid"
            )
        compiled[spec_hash] = object_value
        provenance[spec_hash] = object_provenance
    expected_comparison_keys = set(case.exception_semantic_keys)
    if any(not _is_sha256(key) for key in expected_comparison_keys):
        raise RuntimeError(f"{case.name}: exception comparison key is invalid")
    observed_comparison_keys = {
        str(item["comparison_semantic_key"]) for item in provenance.values()
    }
    if not expected_comparison_keys <= observed_comparison_keys:
        raise RuntimeError(
            f"{case.name}: expected exception comparison keys are not bound to "
            f"the exact arm objects: expected={sorted(expected_comparison_keys)}, "
            f"observed={sorted(observed_comparison_keys)}"
        )
    return compiled, provenance


def _validate_arm_pair(
    case: CaseContract,
    a: Mapping[str, Mapping[str, object]],
    b: Mapping[str, Mapping[str, object]],
) -> None:
    if set(a) != set(case.spec_hashes) or set(b) != set(case.spec_hashes):
        raise RuntimeError(f"{case.name}: artifact bundle does not match case specs")
    for spec_hash in case.spec_hashes:
        for field in (
            "compile_spec_hash",
            "compile_spec_json",
            "comparison_semantic_key",
            "kernel_id",
            "package_fingerprint",
            "target",
            "launch_metadata",
        ):
            if a[spec_hash].get(field) != b[spec_hash].get(field):
                raise RuntimeError(f"{case.name}/{spec_hash}: arm {field} differs")


def _verify_immutable_artifact(
    provenance: Mapping[str, object],
) -> dict[str, object]:
    verified = verify_artifact(provenance)
    ptx_evidence = provenance.get("ptx_evidence")
    if not isinstance(ptx_evidence, Mapping):
        raise RuntimeError("artifact provenance is missing PTX evidence")
    sidecar_path = Path(str(ptx_evidence["path"]))
    sidecar_sha256 = sha256_file(sidecar_path)
    if sidecar_sha256 != ptx_evidence["sha256"]:
        raise RuntimeError(f"PTX evidence changed during benchmark: {sidecar_path}")
    return {**verified, "ptx_evidence_sha256": sidecar_sha256}


@contextmanager
def _exact_compile_resolution(
    module: ModuleType,
    compiled_by_spec: Mapping[str, object],
    observed_specs: list[str],
):
    original = module.sparkinfer_compile

    def resolve_exact(*_args, compile_spec=None, **_kwargs):
        if compile_spec is None:
            raise RuntimeError("production resolver requested a compile without a spec")
        spec_hash = str(compile_spec.hash_key)
        observed_specs.append(spec_hash)
        try:
            return compiled_by_spec[spec_hash]
        except KeyError as exc:
            raise RuntimeError(
                f"production resolver requested unpinned spec {spec_hash}; "
                f"allowed={sorted(compiled_by_spec)}"
            ) from exc

    module.sparkinfer_compile = resolve_exact
    try:
        yield
    finally:
        module.sparkinfer_compile = original


def _dense_cache_clear() -> None:
    import sparkinfer.gemm.dense as dense

    dense._get_compiled_dense_gemm.cache_clear()
    dense._get_compiled_dense_gemm_fused_quant_a.cache_clear()
    dense._get_compiled_dense_gemm_fused_quant_a_grouped.cache_clear()


def _build_dense_nvfp4() -> CaseState:
    import sparkinfer.gemm.dense as dense
    from tests.test_cute_migration_gemm_corpus import (
        _dequantize_nvfp4_dense_operand,
        _quantize_nvfp4_operand,
    )

    generator = torch.Generator(device="cuda").manual_seed(46_001)
    m, n, k = 32, 128, 128
    sources = tuple(
        (
            torch.randn(
                (1, m, k),
                generator=generator,
                dtype=torch.bfloat16,
                device="cuda",
            )
            / 4
        )
        for _ in range(2)
    )
    b_source = (
        torch.randn(
            (1, n, k),
            generator=generator,
            dtype=torch.bfloat16,
            device="cuda",
        )
        / 4
    )
    quantized = tuple(_quantize_nvfp4_operand(source) for source in sources)
    operands = tuple(operand for operand, _ in quantized)
    a_globals = tuple(global_scale for _, global_scale in quantized)
    b, b_global = _quantize_nvfp4_operand(b_source)
    b_dequant = _dequantize_nvfp4_dense_operand(b, k=k, global_scale=b_global)
    references = tuple(
        torch.einsum(
            "gmk,gnk->mng",
            _dequantize_nvfp4_dense_operand(operand, k=k, global_scale=global_scale),
            b_dequant,
        ).to(torch.bfloat16)
        for operand, global_scale in zip(operands, a_globals, strict=True)
    )
    live_values = operands[0][0].clone()
    live_scales = operands[0][1].clone()
    live_alpha = (1.0 / (a_globals[0][0] * b_global[0])).reshape(1).clone()
    scenario_alpha = tuple(
        (1.0 / (global_scale[0] * b_global[0])).reshape(1) for global_scale in a_globals
    )
    output = torch.empty((m, n, 1), dtype=torch.bfloat16, device="cuda")

    def install(index: int) -> None:
        live_values.copy_(operands[index][0])
        live_scales.copy_(operands[index][1])
        live_alpha.copy_(scenario_alpha[index])

    def launch() -> torch.Tensor:
        return dense.dense_gemm(
            (live_values, live_scales),
            b,
            out=output,
            alpha=live_alpha,
            ab_dtype="float4_e2m1fn",
            sf_dtype="float8_e4m3fn",
            c_dtype="bfloat16",
            sf_vec_size=16,
            mma_tiler_mn=(64, 64),
            load_path="tma",
            swap_ab=False,
        )

    return CaseState(
        launch=launch,
        install=install,
        output=output,
        references=references,
        resolver_module=dense,
        clear_resolution_cache=_dense_cache_clear,
        stable_tensors={
            "a_values": live_values,
            "a_scales": live_scales,
            "alpha": live_alpha,
            "b_values": b[0],
            "b_scales": b[1],
            "output": output,
        },
        read_only_tensors={"b_values": b[0], "b_scales": b[1]},
        workspace={"kind": "caller-owned-output/no-extra-scratch"},
        owners=(sources, quantized, operands, a_globals, b, b_global, references),
    )


def _build_dense_fused(*, grouped: bool) -> CaseState:
    import sparkinfer.gemm.dense as dense
    from sparkinfer.gemm.wo_projection import quantize_mxfp8_rows_torch
    from tests.test_cute_migration_gemm_corpus import _mxfp8_gemm_reference

    generator = torch.Generator(device="cuda").manual_seed(
        46_003 if grouped else 46_002
    )
    m, n, k, groups = 2, 128, 128, (2 if grouped else 1)
    source_shape = (m, groups, k) if grouped else (m, k)
    scenarios = tuple(
        (
            torch.randn(
                source_shape,
                generator=generator,
                dtype=torch.bfloat16,
                device="cuda",
            )
            / 4
        ).contiguous()
        for _ in range(2)
    )
    b_source = (
        torch.randn(
            (n, k, groups),
            generator=generator,
            dtype=torch.bfloat16,
            device="cuda",
        )
        / 32
    ).contiguous()
    b_quant = quantize_mxfp8_rows_torch(b_source)
    references = tuple(
        _mxfp8_gemm_reference(
            source.permute(0, 2, 1) if grouped else source.unsqueeze(-1),
            b_quant.values,
            b_quant.scale_rows,
        )
        for source in scenarios
    )
    live_source = scenarios[0].clone()
    if grouped:
        output = torch.empty(
            (groups, m, n), dtype=torch.bfloat16, device="cuda"
        ).as_strided((m, n, groups), (n, 1, m * n))
    else:
        output = torch.empty((m, n, 1), dtype=torch.bfloat16, device="cuda")

    def install(index: int) -> None:
        live_source.copy_(scenarios[index])

    if grouped:

        def launch() -> torch.Tensor:
            return dense.dense_gemm_fused_quant_a_grouped(
                live_source,
                b_quant.values,
                b_quant.scale_mma,
                groups=groups,
                out=output,
                mma_tiler_mn=(64, 64),
            )

    else:

        def launch() -> torch.Tensor:
            return dense.dense_gemm_fused_quant_a(
                live_source,
                b_quant.values,
                b_quant.scale_mma,
                out=output,
                mma_tiler_mn=(64, 64),
            )

    return CaseState(
        launch=launch,
        install=install,
        output=output,
        references=references,
        resolver_module=dense,
        clear_resolution_cache=_dense_cache_clear,
        stable_tensors={
            "source": live_source,
            "b_values": b_quant.values,
            "b_scale_mma": b_quant.scale_mma,
            "output": output,
        },
        read_only_tensors={
            "b_values": b_quant.values,
            "b_scale_rows": b_quant.scale_rows,
            "b_scale_mma": b_quant.scale_mma,
        },
        workspace={"kind": "caller-owned-output/no-split-k-scratch"},
        owners=(scenarios, b_source, b_quant, references),
    )


def _tp_cache_clear() -> None:
    import sparkinfer.integration.tp_moe as tp_moe

    tp_moe.clear_tp_moe_caches()
    tp_moe._TINY_DECODE_KERNEL_CACHE.clear()


def _build_tp_moe(*, tiny: bool) -> CaseState:
    import sparkinfer.integration.tp_moe as tp_moe
    from tests.test_cute_migration_moe_standard_corpus import (
        _make_inputs,
        _make_mxfp4_weights,
        _make_nvfp4_weights,
        _mxfp4_oracle,
        _nvfp4_oracle,
        _prepare_and_bind,
    )

    device = torch.device("cuda")
    if tiny:
        weights = _make_mxfp4_weights(device, seed=301)
        templates = (
            _make_inputs(device, m=2, seed=302, route_shift=0),
            _make_inputs(device, m=2, seed=303, route_shift=2),
        )
        references = tuple(_mxfp4_oracle(weights, value) for value in templates)
        quant_mode = "w4a8_mx"
        source_format = "fp4_e8m0_k32"
    else:
        weights = _make_nvfp4_weights(device, seed=101)
        templates = (
            _make_inputs(device, m=2, seed=102, route_shift=0),
            _make_inputs(device, m=2, seed=103, route_shift=2),
        )
        references = tuple(
            _nvfp4_oracle(weights, value, quant_scale_math="reciprocal_multiply")
            for value in templates
        )
        quant_mode = "nvfp4"
        source_format = "modelopt_nvfp4"
    live = templates[0]
    bound = _prepare_and_bind(
        weights,
        live,
        quant_mode=quant_mode,
        source_format=source_format,
    )
    output = bound.binding.output
    if output is None:
        raise AssertionError("TP-MoE binding has no caller-owned output")
    if not bound.scratch_plan.caps.frozen:
        raise AssertionError("TP-MoE scratch plan is not frozen")

    def install(index: int) -> None:
        template = templates[index]
        live.a.copy_(template.a)
        live.topk_ids.copy_(template.topk_ids)
        live.topk_weights.copy_(template.topk_weights)

    def launch() -> torch.Tensor:
        result = tp_moe.sparkinfer_moe_fp4(binding=bound.binding)
        if result.data_ptr() != output.data_ptr():
            raise AssertionError("TP-MoE production route replaced bound output")
        return result

    scratch_specs = tuple(bound.scratch_plan.scratch_specs())
    return CaseState(
        launch=launch,
        install=install,
        output=output,
        references=references,
        resolver_module=tp_moe,
        clear_resolution_cache=_tp_cache_clear,
        stable_tensors={
            "activations": live.a,
            "topk_ids": live.topk_ids,
            "topk_weights": live.topk_weights,
            "output": output,
            **{
                f"scratch.{index}": tensor for index, tensor in enumerate(bound.scratch)
            },
        },
        read_only_tensors={
            f"reference.{index}": value for index, value in enumerate(references)
        },
        workspace={
            "kind": "TPMoEScratchPlan",
            "frozen": True,
            "implementation": bound.scratch_plan.launch_plan.implementation,
            "scratch_specs": [
                {
                    "shape": list(spec.shape),
                    "dtype": str(spec.dtype),
                    "device": str(spec.device),
                }
                for spec in scratch_specs
            ],
        },
        owners=(weights, templates, bound, references),
    )


def _w4_cache_clear() -> None:
    import sparkinfer.moe.fused.w4a16.kernel as w4a16_kernel

    w4a16_kernel.clear_w4a16_kernel_cache()


def _build_w4a16_standalone() -> CaseState:
    from validation.cutlass_migration.diagnostics.w4a16_standalone import (
        _gemm_reference,
        _make_source_weights,
    )
    from sparkinfer.cute.utils import current_cuda_stream
    from sparkinfer.moe.fused.w4a16.host import packed_gemm_scratch_elements
    import sparkinfer.moe.fused.w4a16.kernel as w4a16_kernel
    from sparkinfer.moe.fused.w4a16.prepare import (
        prepare_w4a16_modelopt_nvfp4_weights,
    )

    device = torch.device("cuda")
    m, hidden, intermediate, experts, topk = 128, 128, 128, 8, 4
    activation = "silu"
    block_size, tile_n, tile_k = 64, 128, 64
    source_weights = _make_source_weights(
        experts=experts,
        hidden_size=hidden,
        intermediate_size=intermediate,
        activation=activation,
        device=device,
    )
    prepared = prepare_w4a16_modelopt_nvfp4_weights(
        *source_weights, activation=activation, params_dtype=torch.bfloat16
    )
    generator = torch.Generator(device="cuda").manual_seed(20260718)
    scenarios = tuple(
        (
            torch.randn(
                (m, hidden), generator=generator, device=device, dtype=torch.float32
            )
            * 0.25
        ).to(torch.bfloat16)
        for _ in range(2)
    )
    live_x = scenarios[0].clone()
    token = torch.arange(m, device=device, dtype=torch.int32)[:, None]
    rank = torch.arange(topk, device=device, dtype=torch.int32)[None, :]
    topk_ids = ((token + rank * 3) % experts).contiguous()
    topk_weights = torch.full((m, topk), 1.0 / topk, dtype=torch.float32, device=device)
    packed_routes, block_experts, packed_route_count = (
        w4a16_kernel.pack_topk_routes_by_expert(topk_ids, block_size, experts)
    )
    torch.cuda.synchronize()
    active_slots = int(packed_route_count.item())
    active_blocks = active_slots // block_size
    rows = 2 * intermediate
    references = tuple(
        _gemm_reference(
            value,
            topk_ids,
            source_weights[0],
            source_weights[1],
            source_weights[2],
            intermediate_size=intermediate,
            activation=activation,
        )
        for value in scenarios
    )
    output = torch.empty((m * topk, rows), dtype=torch.bfloat16, device=device)
    props = torch.cuda.get_device_properties(device)
    sms = int(props.multi_processor_count)
    c_tmp = torch.empty(
        packed_gemm_scratch_elements(
            size_n=rows,
            route_slots=int(packed_routes.numel()),
            moe_block_size=block_size,
            sms=sms,
        ),
        dtype=torch.float32,
        device=device,
    )
    locks = torch.zeros(sms * 4, dtype=torch.int32, device=device)

    def install(index: int) -> None:
        live_x.copy_(scenarios[index])

    def launch() -> torch.Tensor:
        gemm = w4a16_kernel.compile_w4a16_gemm(
            size_m=m,
            size_n=rows,
            size_k=hidden,
            num_experts=experts,
            top_k=topk,
            mul_topk_weights=False,
            tile_n=tile_n,
            tile_k=tile_k,
            moe_block_size=block_size,
            max_m_blocks=int(block_experts.numel()),
            element_dtype="bf16",
            scale_format="e4m3_k16",
        )
        grid_x = max(
            min(sms * int(gemm.blocks_per_sm), active_blocks * (rows // tile_n)),
            1,
        )
        gemm.compiled(
            live_x.reshape(-1),
            prepared.w13.reshape(-1),
            output.reshape(-1),
            prepared.w13_scale.view(torch.uint8).view(torch.int32).reshape(-1),
            prepared.w13_global_scale.reshape(-1),
            packed_routes.reshape(-1),
            block_experts.reshape(-1),
            packed_route_count.reshape(-1),
            topk_weights.reshape(-1),
            c_tmp.reshape(-1),
            locks.reshape(-1),
            m,
            grid_x,
            current_cuda_stream(),
        )
        return output

    return CaseState(
        launch=launch,
        install=install,
        output=output,
        references=references,
        resolver_module=w4a16_kernel,
        clear_resolution_cache=_w4_cache_clear,
        stable_tensors={
            "x": live_x,
            "topk_ids": topk_ids,
            "topk_weights": topk_weights,
            "packed_routes": packed_routes,
            "block_experts": block_experts,
            "packed_route_count": packed_route_count,
            "c_tmp": c_tmp,
            "locks": locks,
            "output": output,
        },
        read_only_tensors={
            "prepared_w13": prepared.w13,
            "prepared_w13_scale": prepared.w13_scale,
            "prepared_w13_global": prepared.w13_global_scale,
        },
        workspace={
            "kind": "standalone-packed-route-gemm",
            "route_capacity_slots": int(packed_routes.numel()),
            "active_route_slots": active_slots,
            "active_route_blocks": active_blocks,
            "c_tmp_elements": int(c_tmp.numel()),
            "lock_elements": int(locks.numel()),
        },
        owners=(source_weights, prepared, scenarios, references),
    )


def _build_w4a16_small_m() -> CaseState:
    from sparkinfer.moe.fused.reference import moe_reference_w4a16_fp4_e8m0_k32
    import sparkinfer.moe.fused.w4a16.kernel as w4a16_kernel
    from sparkinfer.moe.fused.w4a16.prepare import (
        make_w4a16_packed_buffers,
        prepare_w4a16_e8m0_native_weights,
    )
    from tests.test_w4a16_e2e import _pattern_e8m0

    device = torch.device("cuda")
    m, hidden, intermediate, experts, topk = 1, 128, 128, 4, 2
    rows = 2 * intermediate
    generator = torch.Generator(device="cuda").manual_seed(20261602)
    w13 = torch.randint(
        0,
        256,
        (experts, rows, hidden // 2),
        dtype=torch.uint8,
        device=device,
        generator=generator,
    )
    w2 = torch.randint(
        0,
        256,
        (experts, hidden, intermediate // 2),
        dtype=torch.uint8,
        device=device,
        generator=generator,
    )
    w13_scale = _pattern_e8m0((experts, rows, hidden // 32))
    w2_scale = _pattern_e8m0((experts, hidden, intermediate // 32), offset=1)
    w13_global = torch.ones(experts, dtype=torch.float32, device=device)
    w2_global = torch.ones(experts, dtype=torch.float32, device=device)
    scenarios = tuple(
        (
            (
                torch.randn(
                    (m, hidden),
                    generator=generator,
                    dtype=torch.float32,
                    device=device,
                )
                * 0.25
            ).to(torch.bfloat16),
            torch.tensor(
                [[index % experts, (index + 1) % experts]],
                dtype=torch.int32,
                device=device,
            ),
            torch.tensor(
                [[0.625 - 0.125 * index, 0.375 + 0.125 * index]],
                dtype=torch.float32,
                device=device,
            ),
        )
        for index in range(2)
    )
    references = tuple(
        moe_reference_w4a16_fp4_e8m0_k32(
            x,
            w13,
            w13_scale,
            w13_global,
            w2,
            w2_scale,
            w2_global,
            ids,
            weights,
            experts,
            hidden,
            intermediate,
            activation="silu",
            swiglu_limit=None,
            w13_layout="w13",
        )
        for x, ids, weights in scenarios
    )
    prepared = prepare_w4a16_e8m0_native_weights(
        w13,
        w13_scale,
        w13_global,
        w2,
        w2_scale,
        w2_global,
        activation="silu",
        params_dtype=torch.bfloat16,
        w13_layout="w13",
    )
    buffers = make_w4a16_packed_buffers(
        prepared, m=m, topk=topk, dtype=torch.bfloat16, device=device
    )
    fc2_n_chunks = ((intermediate // 2) + 127) // 128
    intermediate_cache2 = torch.zeros(
        2 * m * fc2_n_chunks * 128 * topk,
        dtype=torch.bfloat16,
        device=device,
    )
    live_x = scenarios[0][0].clone()
    live_ids = scenarios[0][1].clone()
    live_weights = scenarios[0][2].clone()

    def install(index: int) -> None:
        x, ids, weights = scenarios[index]
        live_x.copy_(x)
        live_ids.copy_(ids)
        live_weights.copy_(weights)

    def launch() -> torch.Tensor:
        return w4a16_kernel.run_w4a16_moe(
            live_x,
            prepared,
            live_weights,
            live_ids,
            activation="silu",
            fast_math=True,
            intermediate_cache13=buffers.intermediate_cache13,
            intermediate_cache2=intermediate_cache2,
            output=buffers.output,
            fc1_c_tmp=buffers.fc1_c_tmp,
            fc2_c_tmp=buffers.fc2_c_tmp,
            packed_route_indices=buffers.packed_route_indices,
            block_expert_ids=buffers.block_expert_ids,
            packed_route_count=buffers.packed_route_count,
            expert_offsets=buffers.expert_offsets,
        )

    buffer_tensors = {
        name: value
        for name, value in vars(buffers).items()
        if isinstance(value, torch.Tensor)
    }
    return CaseState(
        launch=launch,
        install=install,
        output=buffers.output,
        references=references,
        resolver_module=w4a16_kernel,
        clear_resolution_cache=_w4_cache_clear,
        stable_tensors={
            "x": live_x,
            "topk_ids": live_ids,
            "topk_weights": live_weights,
            "intermediate_cache2_micro": intermediate_cache2,
            **{f"buffers.{name}": value for name, value in buffer_tensors.items()},
        },
        read_only_tensors={
            "w13": w13,
            "w13_scale": w13_scale,
            "w2": w2,
            "w2_scale": w2_scale,
        },
        workspace={
            "kind": "native-e8m0-small-m-preplanned-buffers",
            "intermediate_cache2_elements": int(intermediate_cache2.numel()),
            "buffer_shapes": {
                name: list(value.shape) for name, value in buffer_tensors.items()
            },
        },
        owners=(
            w13_global,
            w2_global,
            scenarios,
            references,
            prepared,
            buffers,
        ),
    )


def _build_w4a16_swiglu_limit() -> CaseState:
    import sparkinfer.moe.fused.w4a16.kernel as w4a16_kernel
    from sparkinfer.moe.fused.w4a16.prepare import (
        make_w4a16_packed_buffers,
        prepare_w4a16_modelopt_nvfp4_weights,
    )
    from tests.test_w4a16_e2e import _make_weights, _reference_w4a16

    device = torch.device("cuda")
    m, hidden, intermediate, experts, topk = 24, 128, 128, 8, 2
    activation = "silu"
    swiglu_limit = 10.0
    torch.manual_seed(20260519)
    source_weights = _make_weights(
        experts=experts,
        hidden_size=hidden,
        intermediate_size=intermediate,
        activation=activation,
    )
    w13, w13_scale, _, w2, w2_scale, w2_global = source_weights
    w13_global = torch.full((experts,), 8.0, dtype=torch.float32, device=device)
    source_weights = (
        w13,
        w13_scale,
        w13_global,
        w2,
        w2_scale,
        w2_global,
    )
    prepared = prepare_w4a16_modelopt_nvfp4_weights(
        *source_weights,
        activation=activation,
        params_dtype=torch.bfloat16,
    )
    generator = torch.Generator(device="cuda").manual_seed(20260719)
    x0 = (
        torch.randn(
            (m, hidden), generator=generator, device=device, dtype=torch.float32
        )
        * 2.0
    ).to(torch.bfloat16)
    token = torch.arange(m, device=device, dtype=torch.int32)[:, None]
    rank = torch.arange(topk, device=device, dtype=torch.int32)[None, :]
    ids0 = ((token + rank * 3) % experts).contiguous()
    weights0 = torch.softmax(
        torch.randn((m, topk), generator=generator, device=device, dtype=torch.float32),
        dim=-1,
    )
    scenarios = (
        (x0, ids0, weights0),
        (
            x0.neg().contiguous(),
            ((ids0 + 1) % experts).contiguous(),
            (weights0 * 0.5).contiguous(),
        ),
    )
    references = tuple(
        _reference_w4a16(
            x,
            *source_weights,
            ids,
            weights,
            activation=activation,
            swiglu_limit=swiglu_limit,
        )
        for x, ids, weights in scenarios
    )
    buffers = make_w4a16_packed_buffers(
        prepared,
        m=m,
        topk=topk,
        dtype=torch.bfloat16,
        device=device,
    )
    live_x = scenarios[0][0].clone()
    live_ids = scenarios[0][1].clone()
    live_weights = scenarios[0][2].clone()

    def install(index: int) -> None:
        x, ids, weights = scenarios[index]
        live_x.copy_(x)
        live_ids.copy_(ids)
        live_weights.copy_(weights)

    def launch() -> torch.Tensor:
        return w4a16_kernel.run_w4a16_moe(
            live_x,
            prepared,
            live_weights,
            live_ids,
            activation=activation,
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
            swiglu_limit=swiglu_limit,
        )

    buffer_tensors = {
        name: value
        for name, value in vars(buffers).items()
        if isinstance(value, torch.Tensor)
    }
    return CaseState(
        launch=launch,
        install=install,
        output=buffers.output,
        references=references,
        resolver_module=w4a16_kernel,
        clear_resolution_cache=_w4_cache_clear,
        stable_tensors={
            "x": live_x,
            "topk_ids": live_ids,
            "topk_weights": live_weights,
            **{f"buffers.{name}": value for name, value in buffer_tensors.items()},
        },
        read_only_tensors={
            "prepared.w13": prepared.w13,
            "prepared.w13_scale": prepared.w13_scale,
            "prepared.w13_global_scale": prepared.w13_global_scale,
            "prepared.w2": prepared.w2,
            "prepared.w2_scale": prepared.w2_scale,
            "prepared.w2_global_scale": prepared.w2_global_scale,
        },
        workspace={
            "kind": "swiglu-limit-preplanned-buffers",
            "buffer_shapes": {
                name: list(value.shape) for name, value in buffer_tensors.items()
            },
        },
        owners=(source_weights, prepared, buffers, scenarios, references),
    )


def _build_state(case: CaseContract) -> CaseState:
    if case.name == "dense-nvfp4-m32":
        return _build_dense_nvfp4()
    if case.name == "dense-fused-quant-m2":
        return _build_dense_fused(grouped=False)
    if case.name == "dense-grouped-fused-quant-m2-g2":
        return _build_dense_fused(grouped=True)
    if case.name == "tp-moe-nvfp4-micro-m2":
        return _build_tp_moe(tiny=False)
    if case.name == "tp-moe-w4a8-mx-tiny-m2":
        return _build_tp_moe(tiny=True)
    if case.name == "w4a16-standalone-gemm-m128":
        return _build_w4a16_standalone()
    if case.name == "w4a16-native-e8m0-small-m1":
        return _build_w4a16_small_m()
    if case.name == "w4a16-swiglu-limit-m24":
        return _build_w4a16_swiglu_limit()
    raise AssertionError(f"unsupported case {case.name!r}")


def _correctness(
    case: CaseContract,
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> dict[str, object]:
    actual_f32 = actual.float()
    expected_f32 = expected.float()
    difference = actual_f32 - expected_f32
    finite = bool(torch.isfinite(actual_f32).all().item())
    nonzero = int(torch.count_nonzero(actual_f32).item())
    expected_rms = float(expected_f32.square().mean().sqrt().item())
    actual_flat = actual_f32.reshape(-1)
    expected_flat = expected_f32.reshape(-1)
    cosine = float(
        (
            (actual_flat * expected_flat).sum()
            / (actual_flat.norm() * expected_flat.norm()).clamp_min(1.0e-30)
        ).item()
    )
    rmse = float(difference.square().mean().sqrt().item())
    normalized_rmse = rmse / max(expected_rms, 1.0e-30)
    bit_exact = torch.equal(actual, expected)
    passed = bool(
        finite
        and nonzero > 0
        and expected_rms > 1.0e-6
        and (
            bit_exact
            if case.exact_oracle
            else cosine >= case.min_cosine
            and normalized_rmse <= case.max_normalized_rmse
        )
    )
    result = {
        "passed": passed,
        "finite": finite,
        "nonzero": nonzero,
        "bit_exact": bit_exact,
        "cosine": cosine,
        "rmse": rmse,
        "normalized_rmse": normalized_rmse,
        "max_abs": float(difference.abs().max().item()),
        "actual_sha256": tensor_sha256(actual),
        "expected_sha256": tensor_sha256(expected),
    }
    if not passed:
        raise AssertionError(f"{case.name}: GPU oracle failure: {result}")
    return result


def _validate_manifest_kernels(
    topology: Mapping[str, object],
    provenance: Mapping[str, Mapping[str, object]],
) -> None:
    graph_kernels = {
        str(node["kernel_name"]): int(node["dynamic_smem_bytes"])
        for node in topology["nodes"]
        if node["type"] == "CU_GRAPH_NODE_TYPE_KERNEL"
    }
    for object_provenance in provenance.values():
        expected = object_provenance["launch_metadata"]["launch_dynamic_smem_bytes"]
        for name, allowed_smem in expected.items():
            if name not in graph_kernels:
                raise AssertionError(f"captured graph is missing exact kernel {name}")
            if graph_kernels[name] not in [int(value) for value in allowed_smem]:
                raise AssertionError(
                    f"{name}: graph SMEM {graph_kernels[name]} not in {allowed_smem}"
                )


def _mode_snapshot(expected_gpu: int) -> dict[str, object]:
    snapshot = nvidia_smi_gpu_mode_snapshot()
    fields = snapshot.get("fields")
    if not snapshot.get("available") or not isinstance(fields, Mapping):
        raise RuntimeError(f"GPU mode snapshot unavailable: {snapshot}")
    if int(str(fields["index"])) != expected_gpu:
        raise RuntimeError(
            f"expected physical GPU {expected_gpu}, observed {fields['index']}"
        )
    return snapshot


def _capture_arm(
    *,
    case: CaseContract,
    state: CaseState,
    compiled: Mapping[str, object],
    provenance: Mapping[str, Mapping[str, object]],
    stream: torch.cuda.Stream,
) -> tuple[torch.cuda.CUDAGraph, dict[str, object]]:
    state.clear_resolution_cache()
    observed_specs: list[str] = []
    state.install(0)
    torch.cuda.synchronize()
    with _exact_compile_resolution(state.resolver_module, compiled, observed_specs):
        with torch.cuda.stream(stream):
            state.output.fill_(math.nan)
            result = state.launch()
        stream.synchronize()
        if result.data_ptr() != state.output.data_ptr():
            raise AssertionError(f"{case.name}: production route replaced output")
        eager_correctness = _correctness(case, state.output, state.references[0])
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph, stream=stream):
            state.launch()
        stream.synchronize()
    if set(observed_specs) != set(case.spec_hashes):
        raise RuntimeError(
            f"{case.name}: production route resolved {observed_specs}, "
            f"expected {list(case.spec_hashes)}"
        )
    topology = graph_topology(graph)
    _validate_manifest_kernels(topology, provenance)
    return graph, {
        "resolved_specs": observed_specs,
        "eager_correctness": eager_correctness,
        "topology": topology,
    }


def _validate_replay(
    *,
    case: CaseContract,
    state: CaseState,
    graph: torch.cuda.CUDAGraph,
    scenario: int,
    stream: torch.cuda.Stream,
) -> tuple[dict[str, object], torch.Tensor]:
    state.install(scenario)
    torch.cuda.synchronize()
    with torch.cuda.stream(stream):
        state.output.fill_(math.nan)
    stream.synchronize()
    before = allocator_counters()
    with torch.cuda.stream(stream):
        graph.replay()
    stream.synchronize()
    after = allocator_counters()
    if before != after:
        raise AssertionError(
            f"{case.name}: replay allocated memory: {before} -> {after}"
        )
    nan_count = int(torch.isnan(state.output.float()).sum().item())
    if nan_count:
        raise AssertionError(
            f"{case.name}: {nan_count} poisoned output elements survived replay"
        )
    metrics = _correctness(case, state.output, state.references[scenario])
    return (
        {
            "scenario": scenario,
            "allocator_before": before,
            "allocator_after": after,
            "zero_replay_allocations": True,
            "full_output_overwrite": True,
            "correctness": metrics,
        },
        state.output.clone(),
    )


def _run_case(
    *,
    case: CaseContract,
    labels: tuple[str, str],
    compiled: Mapping[str, Mapping[str, object]],
    provenance: Mapping[str, Mapping[str, Mapping[str, object]]],
    precondition_cycles: int,
    warmup_cycles: int,
    cycles: int,
    event_batch_cycles: int,
    l2_flush_bytes: int,
    replays_per_reported_sample: int,
    precondition_seconds: float,
    maximum_precondition_seconds: float,
    expected_physical_gpu: int,
    max_sm_clock_delta_mhz: float,
) -> dict[str, object]:
    state = _build_state(case)
    torch.cuda.synchronize()
    stable_pointers = {
        name: value.data_ptr() for name, value in state.stable_tensors.items()
    }
    read_only_before = {
        name: tensor_sha256(value) for name, value in state.read_only_tensors.items()
    }
    artifact_before = {
        label: {
            spec_hash: _verify_immutable_artifact(object_provenance)
            for spec_hash, object_provenance in provenance[label].items()
        }
        for label in labels
    }
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graphs: dict[str, torch.cuda.CUDAGraph] = {}
    captures: dict[str, dict[str, object]] = {}
    for label in labels:
        graphs[label], captures[label] = _capture_arm(
            case=case,
            state=state,
            compiled=compiled[label],
            provenance=provenance[label],
            stream=stream,
        )
    state.clear_resolution_cache()
    if topology_signature(captures[labels[0]]["topology"]) != topology_signature(
        captures[labels[1]]["topology"]
    ):
        raise AssertionError(f"{case.name}: arm graph topologies differ")

    correctness: dict[str, dict[str, object]] = {"scenario_0": {}, "scenario_1": {}}
    outputs: dict[int, dict[str, torch.Tensor]] = {0: {}, 1: {}}
    for scenario in (0, 1):
        for label in labels:
            result, output = _validate_replay(
                case=case,
                state=state,
                graph=graphs[label],
                scenario=scenario,
                stream=stream,
            )
            correctness[f"scenario_{scenario}"][label] = result
            outputs[scenario][label] = output
        if not torch.equal(outputs[scenario][labels[0]], outputs[scenario][labels[1]]):
            delta = (
                outputs[scenario][labels[1]].float()
                - outputs[scenario][labels[0]].float()
            )
            raise AssertionError(
                f"{case.name}/scenario-{scenario}: arms are not bit exact; "
                f"max_abs={float(delta.abs().max().item())}"
            )
    if torch.equal(outputs[0][labels[0]], outputs[1][labels[0]]):
        raise AssertionError(f"{case.name}: live-input mutation did not change output")

    state.install(0)
    torch.cuda.synchronize()
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
    stable_after = {
        name: value.data_ptr() for name, value in state.stable_tensors.items()
    }
    if stable_after != stable_pointers:
        raise AssertionError(f"{case.name}: stable tensor addresses changed")
    read_only_after = {
        name: tensor_sha256(value) for name, value in state.read_only_tensors.items()
    }
    if read_only_after != read_only_before:
        raise AssertionError(f"{case.name}: read-only tensors changed")
    artifact_after = {
        label: {
            spec_hash: _verify_immutable_artifact(object_provenance)
            for spec_hash, object_provenance in provenance[label].items()
        }
        for label in labels
    }
    return {
        "name": case.name,
        "exception_semantic_keys": list(case.exception_semantic_keys),
        "object_specs": list(case.spec_hashes),
        "corpus_nodeid": case.corpus_nodeid,
        "shape": dict(case.shape),
        "object_provenance": provenance,
        "artifact_verification_before": artifact_before,
        "artifact_verification_after": artifact_after,
        "production_resolution": captures,
        "same_address_across_arms": True,
        "stable_tensor_pointers": stable_pointers,
        "fixed_workspace": dict(state.workspace),
        "fixed_workspace_capacity": True,
        "read_only_tensor_sha256": read_only_before,
        "read_only_inputs_unchanged": True,
        "cuda_graph_replay": True,
        "cuda_graph_topology_equal": True,
        "zero_replay_allocations": True,
        "full_output_overwrite": True,
        "arms_bit_exact": True,
        "live_input_mutation_changed_output": True,
        "replays_per_reported_sample": replays_per_reported_sample,
        "correctness": correctness,
        "conditions": conditions,
    }


@contextmanager
def _controlled_dispatch_environment():
    controlled = {
        "SPARKINFER_W4A8_TINY_DECODE": "1",
        "SPARKINFER_W4A16_SMALL_M_DIRECT": "1",
    }
    removed = ("SPARKINFER_MICRO_DYNAMIC_CUTOVER_PAIRS", "SPARKINFER_DYNAMIC_TILE_MN")
    previous = {name: os.environ.get(name) for name in (*controlled, *removed)}
    try:
        os.environ.update(controlled)
        for name in removed:
            os.environ.pop(name, None)
        _tp_cache_clear()
        _w4_cache_clear()
        yield {**controlled, **{name: None for name in removed}}
    finally:
        _tp_cache_clear()
        _w4_cache_clear()
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def main() -> None:
    args = _args()
    if args.list_cases:
        print(json.dumps(_case_manifest(), indent=2, sort_keys=True))
        return
    required = {
        "--a-cache": args.a_cache,
        "--b-cache": args.b_cache,
        "--expected-physical-gpu": args.expected_physical_gpu,
        "--expected-package-fingerprint": args.expected_package_fingerprint,
        "--output": args.output,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(f"missing runtime arguments: {', '.join(missing)}")
    if args.cycles < 500:
        raise ValueError("--cycles must be at least 500")
    if args.precondition_cycles < 0:
        raise ValueError("--precondition-cycles must be non-negative")
    if (
        args.precondition_seconds <= 0.0
        or args.maximum_precondition_seconds < args.precondition_seconds
        or args.max_sm_clock_delta_mhz <= 0.0
    ):
        raise ValueError("duration preconditioning and clock limits are invalid")
    if args.warmup_cycles < 1 or args.event_batch_cycles < 1:
        raise ValueError("warmup and event-batch cycles must be positive")
    if args.tiny_replays_per_reported_sample < 1:
        raise ValueError("tiny replays per reported sample must be positive")
    labels = (args.a_label, args.b_label)
    if labels[0] == labels[1]:
        raise ValueError("arm labels must differ")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != str(args.expected_physical_gpu) or visible not in {"4", "5"}:
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES must name the requested physical GPU 4 or 5"
        )
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("this benchmark requires an SM120 CUDA GPU")
    observed_fingerprint = cute_compiler.sparkinfer_package_fingerprint()
    if observed_fingerprint != args.expected_package_fingerprint:
        raise RuntimeError(
            "host source fingerprint differs from explicit final-source pin: "
            f"{observed_fingerprint} != {args.expected_package_fingerprint}"
        )

    selected = tuple(dict.fromkeys(args.case or tuple(_CASES)))
    initial_mode = _mode_snapshot(int(args.expected_physical_gpu))
    results: list[dict[str, object]] = []
    with _controlled_dispatch_environment() as controlled_environment:
        for name in selected:
            case = _CASES[name]
            compiled_a, provenance_a = _load_case_artifacts(
                args.a_cache,
                case,
                expected_cutlass=args.a_cutlass_version,
                expected_fingerprint=args.expected_package_fingerprint,
            )
            compiled_b, provenance_b = _load_case_artifacts(
                args.b_cache,
                case,
                expected_cutlass=args.b_cutlass_version,
                expected_fingerprint=args.expected_package_fingerprint,
            )
            _validate_arm_pair(case, provenance_a, provenance_b)
            results.append(
                _run_case(
                    case=case,
                    labels=labels,
                    compiled={labels[0]: compiled_a, labels[1]: compiled_b},
                    provenance={
                        labels[0]: provenance_a,
                        labels[1]: provenance_b,
                    },
                    precondition_cycles=args.precondition_cycles,
                    warmup_cycles=args.warmup_cycles,
                    cycles=args.cycles,
                    event_batch_cycles=args.event_batch_cycles,
                    l2_flush_bytes=args.l2_flush_bytes,
                    precondition_seconds=args.precondition_seconds,
                    maximum_precondition_seconds=args.maximum_precondition_seconds,
                    expected_physical_gpu=int(args.expected_physical_gpu),
                    max_sm_clock_delta_mhz=args.max_sm_clock_delta_mhz,
                    replays_per_reported_sample=(
                        args.tiny_replays_per_reported_sample
                        if case.name in _TIMER_LIMITED_CASES
                        else 1
                    ),
                )
            )
    final_mode = _mode_snapshot(int(args.expected_physical_gpu))
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    report = {
        "schema": "sparkinfer.compute_exceptions.cache_abba.v1",
        "command": [sys.executable, *sys.argv],
        "worktree": str(REPO_ROOT),
        "git_head": _git("rev-parse", "HEAD"),
        "git_status_porcelain": _git("status", "--short"),
        "source_sha256": sha256_file(Path(__file__).resolve()),
        "evidence_status": args.evidence_status,
        "host_package_fingerprint": observed_fingerprint,
        "controlled_dispatch_environment": controlled_environment,
        "gpu": {
            "visible_devices": visible,
            "physical_index": args.expected_physical_gpu,
            "logical_index": torch.cuda.current_device(),
            "name": properties.name,
            "uuid": str(getattr(properties, "uuid", "")),
            "sms": properties.multi_processor_count,
            "capability": list(torch.cuda.get_device_capability()),
        },
        "gpu_mode_initial": initial_mode,
        "gpu_mode_final": final_mode,
        "arms": {
            labels[0]: {
                "cache": str(args.a_cache.resolve()),
                "expected_cutlass_dsl": args.a_cutlass_version,
            },
            labels[1]: {
                "cache": str(args.b_cache.resolve()),
                "expected_cutlass_dsl": args.b_cutlass_version,
            },
        },
        "case_manifest": _case_manifest(),
        "cases": results,
        "covered_exception_semantic_keys": sorted(
            key
            for case in (_CASES[name] for name in selected)
            for key in case.exception_semantic_keys
        ),
        "all_correct": True,
        "all_graph_topologies_equal": True,
        "all_zero_replay_allocations": True,
        "all_arms_bit_exact": True,
        "all_evidence_gates_passed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "gpu": report["gpu"],
                "covered_exception_semantic_keys": report[
                    "covered_exception_semantic_keys"
                ],
                "ratios_b_over_a": {
                    result["name"]: {
                        condition: payload["timings"]["ratios_b_over_a"]
                        for condition, payload in result["conditions"].items()
                    }
                    for result in results
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
