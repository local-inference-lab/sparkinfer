#!/usr/bin/env python3
"""Static fail-closed checks for CUTLASS migration one-arm adapters.

This is an evidence-tool self-test, not a CPU kernel acceptance test.  Kernel
correctness and performance remain GPU-only on physical GPUs 4 and 5.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import inspect
from types import ModuleType

import torch

CUDA_INITIALIZED_BEFORE_ADAPTER_IMPORTS = torch.cuda.is_initialized()

from validation.cutlass_migration.acceptance.e2e.index import REQUIRED_FAMILIES
from validation.cutlass_migration.acceptance.e2e.readiness import REGISTRY
from validation.cutlass_migration.acceptance.single_arm import (
    bf16_to_fp4_tma as bf16_to_fp4_adapter,
)
from validation.cutlass_migration.acceptance.single_arm import (
    compute_exceptions as compute_adapter,
)
from validation.cutlass_migration.acceptance.single_arm import (
    contiguous_attention as contiguous_attention_adapter,
)
from validation.cutlass_migration.acceptance.single_arm import (
    mla_decode_merge as mla_decode_adapter,
)
from validation.cutlass_migration.acceptance.single_arm import (
    mla_prefill_mg as mla_prefill_adapter,
)
from validation.cutlass_migration.acceptance.single_arm import (
    nsa_indexer as nsa_indexer_adapter,
)
from validation.cutlass_migration.acceptance.single_arm import (
    paged_attention as paged_attention_adapter,
)
from validation.cutlass_migration.acceptance.single_arm import (
    residual_composite as residual_adapter,
)
from validation.cutlass_migration.acceptance.single_arm import (
    residual_prefill_partial as residual_prefill_adapter,
)
from validation.cutlass_migration.acceptance.single_arm import (
    tp_moe_dynamic as tp_moe_adapter,
)
from validation.cutlass_migration.acceptance.single_arm import (
    w4a16_serving as w4a16_serving_adapter,
)
from validation.cutlass_migration.acceptance.single_arm import (
    w4a16_topk_sum as w4a16_topk_adapter,
)
from validation.cutlass_migration.acceptance.single_arm import (
    w4a8_dynamic as w4a8_adapter,
)
from validation.cutlass_migration.acceptance.single_arm.bf16_to_fp4_tma import (
    CASES as BF16_TO_FP4_CASES,
    CORRECTNESS_GATES as BF16_TO_FP4_GATES,
)
from validation.cutlass_migration.acceptance.single_arm.residual_prefill_partial import (
    CASES as RESIDUAL_PREFILL_CASES,
    CORRECTNESS_GATES as RESIDUAL_PREFILL_GATES,
    TOKENS as RESIDUAL_PREFILL_TOKENS,
)
from validation.cutlass_migration.acceptance.single_arm.w4a16_topk_sum import (
    ARTIFACT_ROLE as W4A16_TOPK_ROLE,
    CASES as W4A16_TOPK_CASES,
    CORRECTNESS_GATES as W4A16_TOPK_GATES,
    DECODE_M,
    DECODE_REPLAYS_PER_REPORTED_SAMPLE,
    KERNEL_ID,
    PREFILL_M,
    PREFILL_REPLAYS_PER_REPORTED_SAMPLE,
)
from validation.cutlass_migration.core.exact_cache_abba import (
    json_sha256,
    time_single_graph_conditions,
)
from validation.cutlass_migration.core.single_arm_e2e import (
    bind_exact_artifact,
    build_exact_launch_plan,
    verify_case_compile_contract,
)

CUDA_INITIALIZED_AFTER_ADAPTER_IMPORTS = torch.cuda.is_initialized()


ADAPTERS: dict[str, ModuleType] = {
    "bf16_to_fp4_tma": bf16_to_fp4_adapter,
    "compute_exceptions": compute_adapter,
    "contiguous_attention": contiguous_attention_adapter,
    "mla_decode_merge": mla_decode_adapter,
    "mla_prefill_mg": mla_prefill_adapter,
    "nsa_indexer": nsa_indexer_adapter,
    "paged_attention": paged_attention_adapter,
    "residual_composite": residual_adapter,
    "residual_prefill_partial": residual_prefill_adapter,
    "tp_moe_dynamic": tp_moe_adapter,
    "w4a16_serving": w4a16_serving_adapter,
    "w4a16_topk_sum": w4a16_topk_adapter,
    "w4a8_dynamic": w4a8_adapter,
}

EXPECTED_CASE_COUNTS = {
    "bf16_to_fp4_tma": 7,
    "compute_exceptions": 8,
    "contiguous_attention": 3,
    "mla_decode_merge": 14,
    "mla_prefill_mg": 77,
    "nsa_indexer": 13,
    "paged_attention": 10,
    "residual_composite": 28,
    "residual_prefill_partial": 3,
    "tp_moe_dynamic": 2,
    "w4a16_serving": 11,
    "w4a16_topk_sum": 11,
    "w4a8_dynamic": 1,
}


def _require_equal(label: str, observed: object, expected: object) -> None:
    if observed != expected:
        raise AssertionError(
            f"{label} changed: expected {expected!r}, got {observed!r}"
        )


def _case_suffix(case_id: str) -> str:
    return case_id.split("/", 1)[1]


def _validate_adapter_identities() -> None:
    if CUDA_INITIALIZED_BEFORE_ADAPTER_IMPORTS:
        raise AssertionError("static adapter self-test started with CUDA initialized")
    if CUDA_INITIALIZED_AFTER_ADAPTER_IMPORTS or torch.cuda.is_initialized():
        raise AssertionError("static adapter imports initialized CUDA")

    required_families = tuple(REQUIRED_FAMILIES)
    _require_equal("registry family order", tuple(REGISTRY), required_families)
    _require_equal("imported adapter family order", tuple(ADAPTERS), required_families)
    _require_equal(
        "expected case-count families",
        tuple(EXPECTED_CASE_COUNTS),
        required_families,
    )
    _require_equal("closed adapter family count", len(ADAPTERS), 13)

    observed_counts: dict[str, int] = {}
    global_case_ids: list[str] = []
    for family, module in ADAPTERS.items():
        _require_equal(f"{family} FAMILY", getattr(module, "FAMILY", None), family)
        binding = REGISTRY[family]
        expected_path = f"{module.__name__.replace('.', '/')}.py"
        _require_equal(
            f"{family} registry adapter path",
            binding.single_arm,
            expected_path,
        )

        cases = tuple(getattr(module, "CASES", ()))
        observed_counts[family] = len(cases)
        expected_schema = getattr(module, "INPUT_SCHEMA", None)
        if not isinstance(expected_schema, str) or not expected_schema:
            raise AssertionError(f"{family} has no valid INPUT_SCHEMA")

        family_case_ids: list[str] = []
        family_input_hashes: list[str] = []
        for index, case in enumerate(cases):
            case_id = getattr(case, "case_id", None)
            if (
                not isinstance(case_id, str)
                or not case_id.startswith(f"{family}/")
                or not _case_suffix(case_id)
                or any(character.isspace() for character in case_id)
            ):
                raise AssertionError(
                    f"{family} case {index} has invalid case_id {case_id!r}"
                )
            contract = getattr(case, "input_contract", None)
            if not isinstance(contract, dict) or not contract:
                raise AssertionError(
                    f"{case_id} input_contract must be a nonempty JSON object"
                )
            _require_equal(
                f"{case_id} input schema",
                contract.get("schema"),
                expected_schema,
            )
            _require_equal(
                f"{case_id} input case_id",
                contract.get("case_id"),
                case_id,
            )
            try:
                input_hash = json_sha256(contract)
            except (TypeError, ValueError) as exc:
                raise AssertionError(
                    f"{case_id} input_contract is not canonical finite JSON"
                ) from exc
            if len(input_hash) != 64 or any(
                character not in "0123456789abcdef" for character in input_hash
            ):
                raise AssertionError(
                    f"{case_id} has invalid input-contract hash {input_hash!r}"
                )
            family_case_ids.append(case_id)
            family_input_hashes.append(input_hash)

        if len(family_case_ids) != len(set(family_case_ids)):
            raise AssertionError(f"{family} case identifiers are not unique")
        if len(family_input_hashes) != len(set(family_input_hashes)):
            raise AssertionError(f"{family} input-contract hashes are not unique")
        global_case_ids.extend(family_case_ids)

    _require_equal("adapter case counts", observed_counts, EXPECTED_CASE_COUNTS)
    _require_equal("closed adapter case total", len(global_case_ids), 188)
    if len(global_case_ids) != len(set(global_case_ids)):
        raise AssertionError("single-arm case identifiers are not globally unique")


def _validate_coverage_matrices() -> None:
    _require_equal(
        "BF16-to-FP4 tile and prefill matrix",
        tuple((case.M, case.K) for case in bf16_to_fp4_adapter.CASES),
        (
            (128, 128),
            (128, 256),
            (128, 4_096),
            (512, 4_096),
            (2_048, 4_096),
            (128, 7_168),
            (512, 7_168),
        ),
    )
    _require_equal(
        "compute-exception boundary matrix",
        tuple(
            (
                _case_suffix(case.case_id),
                case.input_contract["shape"]["m"],
            )
            for case in compute_adapter.CASES
        ),
        (
            ("dense-nvfp4-m32", 32),
            ("dense-fused-quant-m2", 2),
            ("dense-grouped-fused-quant-m2-g2", 2),
            ("tp-moe-nvfp4-micro-m2", 2),
            ("tp-moe-w4a8-mx-tiny-m2", 2),
            ("w4a16-standalone-gemm-m128", 128),
            ("w4a16-native-e8m0-small-m1", 1),
            ("w4a16-swiglu-limit-m24", 24),
        ),
    )
    _require_equal(
        "contiguous-attention specialization matrix",
        tuple(case.name for case in contiguous_attention_adapter.CASES),
        ("fixed", "typed-smem-boundary", "varlen-live-metadata"),
    )

    decode_names = ("dsv4-extra-per-token", "glm", "glm-per-token")
    merge_names = ("merge", "sink-merge")
    _require_equal(
        "MLA decode and merge row matrix",
        tuple(
            (
                case.paired_case.name,
                case.input_contract["shape"]["kind"],
                case.rows,
            )
            for case in mla_decode_adapter.CASES
        ),
        (
            *((name, "decode", rows) for name in decode_names for rows in (1, 2)),
            *(
                (name, "merge", rows)
                for name in merge_names
                for rows in (2, 4, 32, 128)
            ),
        ),
    )

    mla_prefill_names = (
        "dsv4-fp8-hg2-h32-topk512",
        "dsv4-bf16-hg1-h16-topk128-sink",
        "dsv4-bf16-hg1-h16-topk128-no-sink",
        "dsv4-fp8-hg1-h16-topk512",
        "dsv4-bf16-hg2-h32-topk128",
        "dsv4-bf16-hg1-h16-topk128-extra64-sink",
        "glm-fp8-hg1-h16-topk512",
        "glm-fp8-hg2-h32-topk512",
        "glm-nvfp4-bf16-hg1-h16-topk512",
        "glm-nvfp4-bf16-hg2-h32-topk512",
        "glm-fp8-hg1-h8-topk512-packed-hilo",
    )
    mla_prefill_rows = (1, 2, 8, 32, 128, 512, 2_048)
    observed_mla_prefill_names = tuple(
        dict.fromkeys(case.paired_case.name for case in mla_prefill_adapter.CASES)
    )
    _require_equal(
        "MLA prefill specialization order",
        observed_mla_prefill_names,
        mla_prefill_names,
    )
    _require_equal(
        "MLA prefill specialization-by-row matrix",
        tuple(
            (
                name,
                tuple(
                    case.rows
                    for case in mla_prefill_adapter.CASES
                    if case.paired_case.name == name
                ),
            )
            for name in observed_mla_prefill_names
        ),
        tuple((name, mla_prefill_rows) for name in mla_prefill_names),
    )

    _require_equal(
        "NSA indexer decode/prefill and paged matrix",
        tuple(case.name for case in nsa_indexer_adapter.CASES),
        (
            "msa-contiguous-block-scores",
            "contiguous-decode",
            "contiguous-prefill",
            "contiguous-prefill512-h32",
            "contiguous-tiled-prefill",
            "persistent-topk-rows1",
            "persistent-topk-rows2",
            "persistent-topk-rows3",
            "persistent-topk-rows4",
            "paged-base-tiled",
            "paged-scheduled-single",
            "paged-scheduled-multi",
            "paged-stream",
        ),
    )
    _require_equal(
        "paged-attention decode/prefill/verify matrix",
        tuple(
            (
                case.name,
                case.mode,
                case.q_len,
                case.cache_len,
                case.fp8_kv,
                case.disable_split_kv,
            )
            for case in paged_attention_adapter.CASES
        ),
        (
            ("prefill-q8-fp8", "extend", 8, 8, True, False),
            ("prefill-q16-fp8", "extend", 16, 16, True, False),
            ("prefill-q64-fp8", "extend", 64, 64, True, False),
            ("prefill-q128-fp8", "extend", 128, 128, True, False),
            ("prefill-q256-fp8", "extend", 256, 256, True, False),
            ("prefill-q1024-fp8", "extend", 1_024, 1_024, True, False),
            (
                "decode-q1-bf16-direct",
                "decode",
                1,
                256,
                False,
                True,
            ),
            (
                "prefill-q4-bf16-direct-dual-tma-tail",
                "extend",
                4,
                4_096,
                False,
                True,
            ),
            ("verify-q4-bf16-split", "verify", 4, 256, False, False),
            ("verify-q4-fp8-split", "verify", 4, 256, True, False),
        ),
    )

    residual_names = tuple(
        dict.fromkeys(case.paired_case.name for case in residual_adapter.CASES)
    )
    _require_equal(
        "residual decode/prefill matrix",
        tuple(
            (
                name,
                next(
                    case.paired_case.prefill
                    for case in residual_adapter.CASES
                    if case.paired_case.name == name
                ),
                next(
                    case.paired_case.route
                    for case in residual_adapter.CASES
                    if case.paired_case.name == name
                ),
                tuple(
                    case.tokens
                    for case in residual_adapter.CASES
                    if case.paired_case.name == name
                ),
            )
            for name in residual_names
        ),
        (
            ("decode-h4096-split", False, "decode-split", (1,)),
            ("decode-h4096-pre", False, "decode-pre", (1,)),
            ("decode-h7168-split", False, "decode-split", (1,)),
            ("decode-h7168-pre", False, "decode-pre", (1,)),
            (
                "prefill-h4096-bf16-tma",
                True,
                "prefill-bf16-tma",
                (33, 384, 1_024, 2_048),
            ),
            (
                "prefill-h4096-bf16-vector",
                True,
                "prefill-bf16-vector",
                (33, 384, 1_024, 2_048),
            ),
            (
                "prefill-h4096-tf32-tma",
                True,
                "prefill-tf32-tma",
                (33, 384, 1_024, 2_048),
            ),
            (
                "prefill-h7168-bf16-tma",
                True,
                "prefill-bf16-tma",
                (33, 384, 1_024, 2_048),
            ),
            (
                "prefill-h7168-bf16-vector",
                True,
                "prefill-bf16-vector",
                (33, 384, 1_024, 2_048),
            ),
            (
                "prefill-h7168-tf32-tma",
                True,
                "prefill-tf32-tma",
                (33, 384, 1_024, 2_048),
            ),
        ),
    )
    _require_equal(
        "residual prefill-partial matrix",
        tuple(
            (case.kernel_kind, case.hidden_size, residual_prefill_adapter.TOKENS)
            for case in residual_prefill_adapter.CASES
        ),
        (("compact", 4_096, 33), ("compact", 7_168, 33), ("block-m", 7_168, 33)),
    )
    _require_equal(
        "TP-MoE prefill matrix",
        tuple((case.name, case.m, case.quant_mode) for case in tp_moe_adapter.CASES),
        (
            ("nvfp4-prefill-m128", 128, "nvfp4"),
            ("w4a8-mx-materialized-m4096", 4_096, "w4a8_mx"),
        ),
    )
    _require_equal(
        "W4A16 serving decode/prefill matrix",
        tuple(
            (case.serving_regime, case.policy, case.m)
            for case in w4a16_serving_adapter.CASES
        ),
        (
            ("decode", "direct", 1),
            ("decode", "direct", 2),
            ("decode", "direct", 4),
            ("decode", "routed", 8),
            ("decode", "routed", 23),
            ("decode", "routed", 33),
            ("decode", "routed", 80),
            ("prefill", "prefill", 8_192),
            ("prefill", "prefill", 16_384),
            ("prefill", "prefill", 24_576),
            ("prefill", "prefill", 32_768),
        ),
    )
    _require_equal(
        "W4A16 top-k-sum decode/prefill matrix",
        tuple((case.serving_regime, case.m) for case in w4a16_topk_adapter.CASES),
        (
            *(("decode", m) for m in (1, 2, 4, 8, 23, 33, 80)),
            *(("prefill", m) for m in (8_192, 16_384, 24_576, 32_768)),
        ),
    )
    _require_equal(
        "W4A8 dynamic prefill matrix",
        tuple(
            (case.m, case.tile_m, case.recipe, case.activation)
            for case in w4a8_adapter.CASES
        ),
        ((129, 128, "w4a8_nvfp4", "relu2"),),
    )


def _expect_runtime_error(callback, expected: str) -> None:
    try:
        callback()
    except RuntimeError as exc:
        if expected not in str(exc):
            raise AssertionError(
                f"wrong negative-test error: expected {expected!r}, got {str(exc)!r}"
            ) from exc
    else:
        raise AssertionError("invalid adapter contract unexpectedly passed")


def _expect_value_error(callback, expected: str) -> None:
    try:
        callback()
    except ValueError as exc:
        if expected not in str(exc):
            raise AssertionError(
                f"wrong timer failure: expected {expected!r}, got {str(exc)!r}"
            ) from exc
    else:
        raise AssertionError("invalid single-arm timer control unexpectedly passed")


def _validate_required_timer_controls() -> None:
    signature = inspect.signature(time_single_graph_conditions)
    required_controls = (
        "event_batch_replays",
        "precondition_seconds",
        "maximum_precondition_seconds",
        "mode_snapshot",
        "required_pstate",
        "max_sm_clock_delta_mhz",
    )
    for name in required_controls:
        if signature.parameters[name].default is not inspect.Parameter.empty:
            raise AssertionError(
                f"time_single_graph_conditions.{name} must not have a default"
            )

    base = {
        "event_batch_replays": 100,
        "precondition_seconds": 5.0,
        "maximum_precondition_seconds": 30.0,
        "mode_snapshot": lambda: {},
        "required_pstate": "P1",
        "max_sm_clock_delta_mhz": 60.0,
    }
    mutations = (
        ("event_batch_replays", 0, "positive integer"),
        ("precondition_seconds", 4.999, "at least 5 seconds"),
        ("maximum_precondition_seconds", 60.001, "at most 60"),
        ("mode_snapshot", None, "physical-GPU callback"),
        ("required_pstate", "P8", "must be P1"),
        ("max_sm_clock_delta_mhz", 0.0, "must be in (0, 60]"),
    )
    for field, value, expected in mutations:
        controls = {**base, field: value}
        _expect_value_error(
            lambda controls=controls: time_single_graph_conditions(
                None,  # type: ignore[arg-type]
                precondition=2_000,
                warmup=100,
                replays=1_000,
                stream=None,  # validation fails before any CUDA operation
                l2_flush_bytes=0,
                **controls,
            ),
            expected,
        )


def main() -> int:
    _validate_adapter_identities()
    _validate_coverage_matrices()
    _validate_required_timer_controls()
    bf16_shapes = {(case.M, case.K) for case in BF16_TO_FP4_CASES}
    required_bf16_prefill = {
        (128, 4_096),
        (512, 4_096),
        (2_048, 4_096),
        (128, 7_168),
        (512, 7_168),
    }
    if len(BF16_TO_FP4_CASES) != 7 or not required_bf16_prefill <= bf16_shapes:
        raise AssertionError("BF16-to-FP4 prefill matrix is incomplete")
    if not BF16_TO_FP4_GATES or "torch-reference" not in BF16_TO_FP4_GATES:
        raise AssertionError("BF16-to-FP4 correctness gates are incomplete")

    observed_m = tuple(case.m for case in W4A16_TOPK_CASES)
    if observed_m != (*DECODE_M, *PREFILL_M):
        raise AssertionError(f"W4A16 top-k-sum matrix is incomplete: {observed_m}")
    if len(W4A16_TOPK_CASES) != 11 or len(PREFILL_M) != 4:
        raise AssertionError("W4A16 top-k-sum must retain 7 decode + 4 prefill cases")
    if (
        DECODE_REPLAYS_PER_REPORTED_SAMPLE != 64
        or PREFILL_REPLAYS_PER_REPORTED_SAMPLE != 1
    ):
        raise AssertionError("W4A16 aggregate timing defaults changed")
    case_ids = [case.case_id for case in W4A16_TOPK_CASES]
    input_hashes = [json_sha256(case.input_contract) for case in W4A16_TOPK_CASES]
    if len(case_ids) != len(set(case_ids)) or len(input_hashes) != len(
        set(input_hashes)
    ):
        raise AssertionError("W4A16 top-k-sum case/input identities are not unique")
    if not W4A16_TOPK_GATES or "guard-canaries" not in W4A16_TOPK_GATES:
        raise AssertionError("W4A16 top-k-sum correctness gates are incomplete")

    if len(RESIDUAL_PREFILL_CASES) != 3 or RESIDUAL_PREFILL_TOKENS != 33:
        raise AssertionError("residual prefill-partial closed M33 matrix changed")
    residual_kinds = {
        (case.kernel_kind, case.hidden_size) for case in RESIDUAL_PREFILL_CASES
    }
    if residual_kinds != {
        ("compact", 4_096),
        ("compact", 7_168),
        ("block-m", 7_168),
    }:
        raise AssertionError(
            f"residual prefill-partial specialization set changed: {residual_kinds}"
        )
    residual_ids = [case.case_id for case in RESIDUAL_PREFILL_CASES]
    residual_inputs = [
        json_sha256(case.input_contract) for case in RESIDUAL_PREFILL_CASES
    ]
    residual_specs = [case.spec_hash for case in RESIDUAL_PREFILL_CASES]
    if any(
        len(values) != len(set(values))
        for values in (residual_ids, residual_inputs, residual_specs)
    ):
        raise AssertionError("residual prefill-partial identities are not unique")
    if (
        "torch-projection-partials-reference" not in RESIDUAL_PREFILL_GATES
        or "torch-gram-partials-reference" not in RESIDUAL_PREFILL_GATES
        or "guard-canaries" not in RESIDUAL_PREFILL_GATES
    ):
        raise AssertionError(
            "residual prefill-partial correctness gates are incomplete"
        )

    compile_spec_json = '{"kernel":"w4a16-topk","version":1}'
    compile_identity = {
        "role": W4A16_TOPK_ROLE,
        "kernel_id": KERNEL_ID,
        "compile_spec_hash": hashlib.sha256(compile_spec_json.encode()).hexdigest(),
        "compile_spec_json": compile_spec_json,
    }
    compile_contract = {
        "artifacts": [compile_identity],
        "launch_plan": [
            {
                "node_index": 0,
                "artifact_role": W4A16_TOPK_ROLE,
                "kernel_id": KERNEL_ID,
                "compile_spec_hash": compile_identity["compile_spec_hash"],
                "multiplicity_index": 1,
            }
        ],
        "source_owned_kernel_nodes": [],
    }
    reviewed = {"compile_artifact_contract": {"current": compile_contract}}
    verify_case_compile_contract(
        case_id=case_ids[0],
        reviewed=reviewed,
        arm="current",
        role=W4A16_TOPK_ROLE,
        provenance=compile_identity,
    )
    for field in ("kernel_id", "compile_spec_hash", "compile_spec_json"):
        substituted = deepcopy(compile_identity)
        substituted[field] = f"substituted-{field}"
        _expect_runtime_error(
            lambda substituted=substituted: verify_case_compile_contract(
                case_id=case_ids[0],
                reviewed=reviewed,
                arm="current",
                role=W4A16_TOPK_ROLE,
                provenance=substituted,
            ),
            "exact object differs from compile contract",
        )
    _expect_runtime_error(
        lambda: verify_case_compile_contract(
            case_id=case_ids[0],
            reviewed=reviewed,
            arm="current",
            role="unreviewed-role",
            provenance=compile_identity,
        ),
        "is not one reviewed artifact role",
    )
    artifact = bind_exact_artifact(
        role=W4A16_TOPK_ROLE,
        evidence={
            **compile_identity,
            "object_sha256": "a" * 64,
        },
    )
    build_exact_launch_plan(
        case_id=case_ids[0],
        reviewed=reviewed,
        arm="current",
        artifacts=[artifact],
        observed_roles=(W4A16_TOPK_ROLE,),
    )
    mixed_compile_contract = deepcopy(compile_contract)
    mixed_compile_contract["launch_plan"][0]["node_index"] = 2
    mixed_compile_contract["source_owned_kernel_nodes"] = [
        {"node_index": 0},
        {"node_index": 1},
    ]
    mixed_reviewed = {"compile_artifact_contract": {"current": mixed_compile_contract}}
    mixed_plan = build_exact_launch_plan(
        case_id=case_ids[0],
        reviewed=mixed_reviewed,
        arm="current",
        artifacts=[artifact],
        observed_roles=((2, W4A16_TOPK_ROLE),),
    )
    if mixed_plan[0]["node_index"] != 2:
        raise AssertionError("mixed graph exact ordinal was renumbered")
    _expect_runtime_error(
        lambda: build_exact_launch_plan(
            case_id=case_ids[0],
            reviewed=mixed_reviewed,
            arm="current",
            artifacts=[artifact],
            observed_roles=(
                (2, W4A16_TOPK_ROLE),
                (1, W4A16_TOPK_ROLE),
            ),
        ),
        "strictly increasing",
    )
    _expect_runtime_error(
        lambda: build_exact_launch_plan(
            case_id=case_ids[0],
            reviewed=reviewed,
            arm="current",
            artifacts=[artifact],
            observed_roles=(W4A16_TOPK_ROLE, W4A16_TOPK_ROLE),
        ),
        "observed launch order or multiplicity differs from review",
    )

    try:
        time_single_graph_conditions(
            None,  # type: ignore[arg-type]
            precondition=1_999,
            warmup=100,
            replays=1_000,
            stream=None,  # type: ignore[arg-type]
            l2_flush_bytes=0,
            event_batch_replays=100,
            precondition_seconds=5.0,
            maximum_precondition_seconds=60.0,
            mode_snapshot=lambda: {},
            required_pstate="P1",
            max_sm_clock_delta_mhz=60.0,
        )
    except ValueError as exc:
        if "precondition>=2000" not in str(exc):
            raise
    else:
        raise AssertionError("single-arm timing accepted an undersized precondition")
    try:
        time_single_graph_conditions(
            None,  # type: ignore[arg-type]
            precondition=2_000,
            warmup=100,
            replays=1_000,
            stream=None,  # type: ignore[arg-type]
            l2_flush_bytes=0,
            replays_per_reported_sample=0,
            event_batch_replays=100,
            precondition_seconds=5.0,
            maximum_precondition_seconds=60.0,
            mode_snapshot=lambda: {},
            required_pstate="P1",
            max_sm_clock_delta_mhz=60.0,
        )
    except ValueError as exc:
        if "positive integer" not in str(exc):
            raise
    else:
        raise AssertionError("single-arm timing accepted zero inner replays")

    print(
        "status=pass adapters=13 cases=188 family-parity=pass "
        "global-case-id-uniqueness=pass input-contract-hashes=pass "
        "decode-prefill-matrices=pass cuda-initialized=false "
        "negative=kernel-id,spec-hash,spec-json,artifact-role,"
        "launch-multiplicity,mixed-node-order,min-samples,aggregate-replay-count,"
        "required-event-duration-pstate-mode-clock-controls"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
