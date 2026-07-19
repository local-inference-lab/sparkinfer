"""One front door for CUTLASS migration qualification tooling."""

from __future__ import annotations

import runpy
import sys
from typing import Final


_SINGLE_ARM_FAMILIES: Final = (
    "bf16_to_fp4_tma",
    "compute_exceptions",
    "contiguous_attention",
    "mla_decode_merge",
    "mla_prefill_mg",
    "nsa_indexer",
    "paged_attention",
    "residual_composite",
    "residual_prefill_partial",
    "tp_moe_dynamic",
    "w4a16_serving",
    "w4a16_topk_sum",
    "w4a8_dynamic",
)

_COMMANDS: Final[dict[tuple[str, ...], str]] = {
    ("acceptance", "corpus"): "validation.cutlass_migration.acceptance.corpus.run",
    (
        "acceptance",
        "readiness",
    ): "validation.cutlass_migration.acceptance.e2e.readiness",
    (
        "acceptance",
        "source-manifest",
    ): "validation.cutlass_migration.acceptance.e2e.source_manifest",
    (
        "acceptance",
        "discovery",
    ): "validation.cutlass_migration.acceptance.e2e.discovery",
    ("acceptance", "contract"): "validation.cutlass_migration.acceptance.e2e.contract",
    (
        "acceptance",
        "evidence-set",
    ): "validation.cutlass_migration.acceptance.e2e.evidence_set",
    ("acceptance", "index"): "validation.cutlass_migration.acceptance.e2e.index",
    (
        "acceptance",
        "release-index",
    ): "validation.cutlass_migration.acceptance.e2e.release_index",
    (
        "acceptance",
        "paired-contract-audit",
    ): "validation.cutlass_migration.acceptance.e2e.paired_contract_audit",
    ("evidence", "resources"): "validation.cutlass_migration.evidence.kernel_resources",
    (
        "evidence",
        "source-inventory",
    ): "validation.cutlass_migration.evidence.source_inventory",
    ("evidence", "sass"): "validation.cutlass_migration.evidence.sass_register_sets",
    ("evidence", "smem"): "validation.cutlass_migration.evidence.smem_contracts",
    (
        "evidence",
        "specialization-contract",
    ): "validation.cutlass_migration.evidence.specialization_contract",
    (
        "evidence",
        "register-accounting",
    ): "validation.cutlass_migration.evidence.register_accounting",
    (
        "evidence",
        "compare-resources",
    ): "validation.cutlass_migration.evidence.compare_resources",
    (
        "evidence",
        "compare-sass",
    ): "validation.cutlass_migration.evidence.compare_sass_register_sets",
    (
        "evidence",
        "merge-resources",
    ): "validation.cutlass_migration.evidence.merge_resource_reports",
    (
        "evidence",
        "reassemble-positive-deltas",
    ): "validation.cutlass_migration.evidence.reassemble_positive_deltas",
    (
        "diagnostic",
        "graph-abba",
    ): "validation.cutlass_migration.diagnostics.graph_replay_abba",
    (
        "diagnostic",
        "graph-case-contract",
    ): "validation.cutlass_migration.diagnostics.graph_replay_case_contract",
    (
        "diagnostic",
        "w4a8-m32",
    ): "validation.cutlass_migration.diagnostics.w4a8_dynamic_m32",
    (
        "diagnostic",
        "w4a8-fc1-intermediate",
    ): "validation.cutlass_migration.diagnostics.w4a8_fc1_intermediate",
    (
        "diagnostic",
        "w4a8-fc1-raw",
    ): "validation.cutlass_migration.diagnostics.w4a8_fc1_raw",
    (
        "diagnostic",
        "w4a8-relu2-nvfp4",
    ): "validation.cutlass_migration.diagnostics.w4a8_relu2_nvfp4",
    (
        "diagnostic",
        "w4a8-staged-input",
    ): "validation.cutlass_migration.diagnostics.w4a8_staged_input",
    (
        "diagnostic",
        "w4a16-standalone",
    ): "validation.cutlass_migration.diagnostics.w4a16_standalone",
    (
        "integrity-check",
        "end-to-end-index",
    ): "validation.cutlass_migration.integrity_checks.end_to_end_index",
    (
        "integrity-check",
        "release-aggregate",
    ): "validation.cutlass_migration.integrity_checks.release_aggregate",
    (
        "integrity-check",
        "exact-cache-abba",
    ): "validation.cutlass_migration.integrity_checks.exact_cache_abba",
    (
        "integrity-check",
        "evidence-set",
    ): "validation.cutlass_migration.integrity_checks.evidence_set",
    (
        "integrity-check",
        "single-arm-adapters",
    ): "validation.cutlass_migration.integrity_checks.single_arm_adapters",
    (
        "integrity-check",
        "corpus-gpu-execution",
    ): "validation.cutlass_migration.integrity_checks.corpus_gpu_execution",
}

for _family in _SINGLE_ARM_FAMILIES:
    _COMMANDS[("acceptance", "single-arm", _family)] = (
        f"validation.cutlass_migration.acceptance.single_arm.{_family}"
    )
    _COMMANDS[("diagnostic", "paired", _family)] = (
        f"validation.cutlass_migration.diagnostics.paired.{_family}"
    )


def _usage() -> str:
    lines = [
        "usage: python -m validation.cutlass_migration <command> [args...]",
        "",
        "GPU acceptance:",
        "  acceptance corpus|readiness|source-manifest|discovery|contract|evidence-set",
        "  acceptance index|release-index",
        "  acceptance paired-contract-audit",
        "  acceptance single-arm <family>",
        "",
        "Evidence:",
        "  evidence resources|source-inventory|sass|smem|specialization-contract",
        "  evidence register-accounting|compare-resources|compare-sass",
        "  evidence merge-resources|reassemble-positive-deltas",
        "",
        "Non-release diagnostics:",
        "  diagnostic paired <family>",
        "  diagnostic graph-abba|graph-case-contract|w4a8-m32",
        "  diagnostic w4a8-fc1-intermediate|w4a8-fc1-raw|w4a8-relu2-nvfp4",
        "  diagnostic w4a8-staged-input|w4a16-standalone",
        "",
        "Offline infrastructure integrity checks (not kernel acceptance):",
        "  integrity-check end-to-end-index|release-aggregate|exact-cache-abba|evidence-set",
        "  integrity-check single-arm-adapters|corpus-gpu-execution|all",
        "",
        "Families:",
        "  " + " ".join(_SINGLE_ARM_FAMILIES),
        "",
        "Append --help after a leaf command for that tool's arguments.",
    ]
    return "\n".join(lines)


def _run(module: str, forwarded: list[str]) -> None:
    sys.argv = [module, *forwarded]
    try:
        runpy.run_module(module, run_name="__main__", alter_sys=False)
    except SystemExit as exc:
        if exc.code not in (None, 0):
            raise


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args in (["-h"], ["--help"], ["help"]):
        print(_usage())
        return 0

    if args[:2] == ["integrity-check", "all"]:
        if len(args) != 2:
            raise SystemExit("integrity-check all does not accept forwarded arguments")
        for name in (
            "end-to-end-index",
            "release-aggregate",
            "exact-cache-abba",
            "evidence-set",
            "single-arm-adapters",
            "corpus-gpu-execution",
        ):
            _run(_COMMANDS[("integrity-check", name)], [])
        return 0

    for width in (3, 2):
        key = tuple(args[:width])
        module = _COMMANDS.get(key)
        if module is not None:
            _run(module, args[width:])
            return 0

    raise SystemExit(f"unknown command: {' '.join(args)}\n\n{_usage()}")


if __name__ == "__main__":
    raise SystemExit(main())
