#!/usr/bin/env python3
"""Exact-object CUDA-graph ABBA for residual exception composites.

This adapter covers the 17 residual migration-exception objects that are not
the two compact partial kernels measured by
``validation.cutlass_migration.diagnostics.paired.residual_prefill_partial``.
Each case calls the public residual serving API with a planner-owned scratch
capacity and pins every CuTe launch in that composite to an immutable A or B
cache object.  No fallback or isolated diagnostic entrypoint is timed.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterator, Mapping

import torch
import torch.nn.functional as F

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
from validation.cutlass_migration.paths import CORE_ROOT, DATA_ROOT, REPO_ROOT
from b12x.integration import (
    B12XMHCScratchCaps,
    b12x_mhc_post_pre,
    b12x_mhc_pre,
    plan_mhc_scratch,
)
from b12x.integration import residual_kernels


_PREFILL_TOKEN_DEFAULTS = (33, 384, 1024, 2048)
_PREFILL_POLICY_MIN = 384
_SPLIT_K = {4096: 64, 7168: 112}


@dataclass(frozen=True)
class ExceptionRow:
    family: str
    spec_hash: str


@dataclass(frozen=True)
class CaseDefinition:
    name: str
    hidden_size: int
    route: str
    target_specs: tuple[str, ...]
    all_specs: tuple[str, ...]
    environment: tuple[tuple[str, str], ...]

    @property
    def prefill(self) -> bool:
        return self.route.startswith("prefill-")


_ROWS = {
    row.spec_hash: row
    for row in (
        ExceptionRow(
            "integration.residual.mhc_finalize_gram_hidden4096",
            "02eeb949ae299cd1642796744d4b27f68367b42595f186da9584b57d562b6589",
        ),
        ExceptionRow(
            "integration.residual.mhc_finalize_gram_hidden4096",
            "f15b8847199f28fb40ae18ba7cf57a27426f4823edec28acfaf96c4b8d98bb27",
        ),
        ExceptionRow(
            "integration.residual.mhc_finalize_gram_hidden4096_compact",
            "e11561fdc72b927b38e76cd1c6ca4a76ce08fd1549bd41f0f4d2b79452c46f86",
        ),
        ExceptionRow(
            "integration.residual.mhc_finalize_gram_hidden7168",
            "1c29d557710fa4c91132c554d7b11fc602af9e74a09a40cc1c7c48e47a726e40",
        ),
        ExceptionRow(
            "integration.residual.mhc_finalize_gram_hidden7168",
            "d37d63e01146a137ce3c399ea554f8cf5cfb01b094d12fc21e18435f42263d6c",
        ),
        ExceptionRow(
            "integration.residual.mhc_finalize_gram_hidden7168_compact",
            "0637890579875d27652828678921f6e1738008cd1c08aac9795ac4c5931f8df1",
        ),
        ExceptionRow(
            "integration.residual.mhc_post_pre_decode_split_n_hidden4096_s4n6x1",
            "18d36e770f1df35c53e3771bedd2e2c128486ab757ddb8a2471f7c2ee60a73df",
        ),
        ExceptionRow(
            "integration.residual.mhc_post_pre_prefill_gram_hidden4096_threads1024",
            "27040623e93dfcd0a4005d8ae26612af4fe7faf0933839aa3c525dd5b21f7777",
        ),
        ExceptionRow(
            "integration.residual.mhc_post_pre_prefill_gram_hidden7168_threads1024",
            "65b7112820fc1b90ae9404b8d79104fb40e8b5cfc0a7d69cd0429796d6417123",
        ),
        ExceptionRow(
            "integration.residual.mhc_pre_partial_hidden4096_hctile128_all4",
            "e45dd0932e0b37e0f3ed2397e2d0130a1926385b8a5c46d0cd9252868bc88599",
        ),
        ExceptionRow(
            "integration.residual.mhc_pre_partial_hidden7168_hctile128_all4",
            "36bff66ea2b3c32e81902fdc6cd1121cb9e13e23f3e6d76a70c259a07d80d59b",
        ),
        ExceptionRow(
            "integration.residual.mhc_prefill_bf16_project_hidden4096_m16_n8",
            "3c0c644ee75d359e10e7c70de982fc32a412f9baa4fb22ad5f0f8cfad1533b3c",
        ),
        ExceptionRow(
            "integration.residual.mhc_prefill_bf16_project_hidden7168_m16_n8",
            "748e7d037719d9df4b9c0496c9653de6b3c421ec78f6bde67eb97a0d9b349d37",
        ),
        ExceptionRow(
            "integration.residual.mhc_prefill_bf16_project_tma_hidden4096_m128_n16",
            "919332856998a152b758138e7f13cb9504cd39b10c4e1791c4f460fc55de2601",
        ),
        ExceptionRow(
            "integration.residual.mhc_prefill_bf16_project_tma_hidden7168_m128_n16",
            "fd7c3bca8cf76beed085056198c638ea924227553e7bb661eb1412b1a20bea81",
        ),
        ExceptionRow(
            "integration.residual.mhc_prefill_tf32_project_tma_hidden4096_m16_n8",
            "40ea1bbd3b9f1feab1d34acb69494e8778d06c84b650ecedaac33ec5a24c1935",
        ),
        ExceptionRow(
            "integration.residual.mhc_prefill_tf32_project_tma_hidden7168_m16_n8",
            "8fa4f5b4c4812ff76bc881d4cf4103b43e14dda1b78d69b335ad32cadcb8f7a5",
        ),
    )
}


_KERNEL_ID_BY_SPEC = {row.spec_hash: row.family for row in _ROWS.values()}
_KERNEL_ID_BY_SPEC.update(
    {
        # This unchanged helper is required by the H7168 split composite whose
        # finalizer is an exception row.
        "a7804e8293a48176bfe4921df579154e5a360adce9b03f2bab2939fefa647308": (
            "integration.residual.mhc_post_pre_decode_split_n_hidden7168_s4n6x1"
        ),
    }
)


_DECODE_ENV = (
    ("B12X_MHC_PREFILL_MIN_TOKENS", "96"),
    ("B12X_MHC_PREFILL_BF16_MIN_TOKENS", "384"),
    ("B12X_MHC_PREFILL_TF32_MIN_TOKENS", "384"),
    ("B12X_MHC_PREFILL_BF16_MMA", "0"),
    ("B12X_MHC_PREFILL_TF32_MMA", "0"),
    ("B12X_MHC_PREFILL_BLOCK_M", "0"),
    ("B12X_MHC_PREFILL_COMPACT", "1"),
    ("B12X_MHC_DECODE_TILE_N", "6"),
)


def _prefill_env(route: str) -> tuple[tuple[str, str], ...]:
    values = dict(_DECODE_ENV)
    values["B12X_MHC_DECODE_SPLITS"] = "0"
    if route == "prefill-bf16-tma":
        values["B12X_MHC_PREFILL_BF16_MMA"] = "1"
        values["B12X_MHC_PREFILL_BF16_TMA"] = "1"
    elif route == "prefill-bf16-vector":
        values["B12X_MHC_PREFILL_BF16_MMA"] = "1"
        values["B12X_MHC_PREFILL_BF16_TMA"] = "0"
    elif route == "prefill-tf32-tma":
        values["B12X_MHC_PREFILL_TF32_MMA"] = "1"
        values["B12X_MHC_PREFILL_BF16_TMA"] = "1"
    else:
        raise AssertionError(route)
    return tuple(sorted(values.items()))


def _case(
    *,
    name: str,
    hidden_size: int,
    route: str,
    target_specs: tuple[str, ...],
    all_specs: tuple[str, ...] | None = None,
) -> CaseDefinition:
    if route == "decode-split":
        environment = (*_DECODE_ENV, ("B12X_MHC_DECODE_SPLITS", "4"))
    elif route == "decode-pre":
        environment = (*_DECODE_ENV, ("B12X_MHC_DECODE_SPLITS", "0"))
    else:
        environment = _prefill_env(route)
    return CaseDefinition(
        name=name,
        hidden_size=hidden_size,
        route=route,
        target_specs=target_specs,
        all_specs=target_specs if all_specs is None else all_specs,
        environment=environment,
    )


_CASES = {
    case.name: case
    for case in (
        _case(
            name="decode-h4096-split",
            hidden_size=4096,
            route="decode-split",
            target_specs=(
                "18d36e770f1df35c53e3771bedd2e2c128486ab757ddb8a2471f7c2ee60a73df",
                "02eeb949ae299cd1642796744d4b27f68367b42595f186da9584b57d562b6589",
            ),
        ),
        _case(
            name="decode-h4096-pre",
            hidden_size=4096,
            route="decode-pre",
            target_specs=(
                "e45dd0932e0b37e0f3ed2397e2d0130a1926385b8a5c46d0cd9252868bc88599",
                "f15b8847199f28fb40ae18ba7cf57a27426f4823edec28acfaf96c4b8d98bb27",
            ),
        ),
        _case(
            name="decode-h7168-split",
            hidden_size=7168,
            route="decode-split",
            target_specs=(
                "d37d63e01146a137ce3c399ea554f8cf5cfb01b094d12fc21e18435f42263d6c",
            ),
            all_specs=(
                "a7804e8293a48176bfe4921df579154e5a360adce9b03f2bab2939fefa647308",
                "d37d63e01146a137ce3c399ea554f8cf5cfb01b094d12fc21e18435f42263d6c",
            ),
        ),
        _case(
            name="decode-h7168-pre",
            hidden_size=7168,
            route="decode-pre",
            target_specs=(
                "36bff66ea2b3c32e81902fdc6cd1121cb9e13e23f3e6d76a70c259a07d80d59b",
                "1c29d557710fa4c91132c554d7b11fc602af9e74a09a40cc1c7c48e47a726e40",
            ),
        ),
        _case(
            name="prefill-h4096-bf16-tma",
            hidden_size=4096,
            route="prefill-bf16-tma",
            target_specs=(
                "27040623e93dfcd0a4005d8ae26612af4fe7faf0933839aa3c525dd5b21f7777",
                "919332856998a152b758138e7f13cb9504cd39b10c4e1791c4f460fc55de2601",
                "e11561fdc72b927b38e76cd1c6ca4a76ce08fd1549bd41f0f4d2b79452c46f86",
            ),
        ),
        _case(
            name="prefill-h4096-bf16-vector",
            hidden_size=4096,
            route="prefill-bf16-vector",
            target_specs=(
                "27040623e93dfcd0a4005d8ae26612af4fe7faf0933839aa3c525dd5b21f7777",
                "3c0c644ee75d359e10e7c70de982fc32a412f9baa4fb22ad5f0f8cfad1533b3c",
                "e11561fdc72b927b38e76cd1c6ca4a76ce08fd1549bd41f0f4d2b79452c46f86",
            ),
        ),
        _case(
            name="prefill-h4096-tf32-tma",
            hidden_size=4096,
            route="prefill-tf32-tma",
            target_specs=(
                "27040623e93dfcd0a4005d8ae26612af4fe7faf0933839aa3c525dd5b21f7777",
                "40ea1bbd3b9f1feab1d34acb69494e8778d06c84b650ecedaac33ec5a24c1935",
                "e11561fdc72b927b38e76cd1c6ca4a76ce08fd1549bd41f0f4d2b79452c46f86",
            ),
        ),
        _case(
            name="prefill-h7168-bf16-tma",
            hidden_size=7168,
            route="prefill-bf16-tma",
            target_specs=(
                "65b7112820fc1b90ae9404b8d79104fb40e8b5cfc0a7d69cd0429796d6417123",
                "fd7c3bca8cf76beed085056198c638ea924227553e7bb661eb1412b1a20bea81",
                "0637890579875d27652828678921f6e1738008cd1c08aac9795ac4c5931f8df1",
            ),
        ),
        _case(
            name="prefill-h7168-bf16-vector",
            hidden_size=7168,
            route="prefill-bf16-vector",
            target_specs=(
                "65b7112820fc1b90ae9404b8d79104fb40e8b5cfc0a7d69cd0429796d6417123",
                "748e7d037719d9df4b9c0496c9653de6b3c421ec78f6bde67eb97a0d9b349d37",
                "0637890579875d27652828678921f6e1738008cd1c08aac9795ac4c5931f8df1",
            ),
        ),
        _case(
            name="prefill-h7168-tf32-tma",
            hidden_size=7168,
            route="prefill-tf32-tma",
            target_specs=(
                "65b7112820fc1b90ae9404b8d79104fb40e8b5cfc0a7d69cd0429796d6417123",
                "8fa4f5b4c4812ff76bc881d4cf4103b43e14dda1b78d69b335ad32cadcb8f7a5",
                "0637890579875d27652828678921f6e1738008cd1c08aac9795ac4c5931f8df1",
            ),
        ),
    )
}


_TARGET_SPEC_SET = {
    spec_hash for case in _CASES.values() for spec_hash in case.target_specs
}
if set(_ROWS) != _TARGET_SPEC_SET:
    raise AssertionError(
        "residual exception coverage contract drifted: "
        f"cases={sorted(_TARGET_SPEC_SET)}, rows={sorted(_ROWS)}"
    )
if len(_TARGET_SPEC_SET) != 17:
    raise AssertionError(
        f"expected 17 residual exception rows, got {len(_TARGET_SPEC_SET)}"
    )


@dataclass
class RuntimeCase:
    definition: CaseDefinition
    tokens: int
    expected_m: int | None
    launch: Any
    install: Any
    expected: Any
    outputs: tuple[torch.Tensor, ...]
    stable_tensors: Mapping[str, torch.Tensor]
    read_only_tensors: Mapping[str, torch.Tensor]
    scenario_tensors: Mapping[str, torch.Tensor]
    scratch_contract: Mapping[str, object]


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_evidence_status_argument(parser)
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--case", choices=tuple(_CASES))
    parser.add_argument("--a-cache", type=Path)
    parser.add_argument("--a-label", default="cutlass-4.5.2")
    parser.add_argument("--b-cache", type=Path)
    parser.add_argument("--b-label", default="cutlass-4.6.0")
    parser.add_argument(
        "--tokens",
        action="append",
        default=[],
        metavar="N[,N...]",
        help=(
            "live token sizes; defaults to 1 for decode and "
            "33,384,1024,2048 for prefill"
        ),
    )
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
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _coverage_manifest() -> dict[str, object]:
    return {
        "schema": "b12x.residual.composite_exception_coverage.v1",
        "exception_row_count": len(_ROWS),
        "case_count": len(_CASES),
        "excluded_dedicated_rows": [
            {
                "family": (
                    "integration.residual."
                    f"mhc_post_pre_prefill_partial_hidden{hidden}_threads512"
                ),
                "reason": (
                    "validation.cutlass_migration.diagnostics.paired."
                    "residual_prefill_partial compact coverage"
                ),
            }
            for hidden in (4096, 7168)
        ],
        "cases": {
            name: {
                "hidden_size": case.hidden_size,
                "route": case.route,
                "production_entrypoint": (
                    "b12x_mhc_pre"
                    if case.route == "decode-pre"
                    else "b12x_mhc_post_pre"
                ),
                "target_exception_rows": [
                    {
                        "family": _ROWS[spec_hash].family,
                        "compile_spec_hash": spec_hash,
                    }
                    for spec_hash in case.target_specs
                ],
                "all_graph_objects": [
                    {
                        "kernel_id": _KERNEL_ID_BY_SPEC[spec_hash],
                        "compile_spec_hash": spec_hash,
                        "exception_row": spec_hash in _ROWS,
                    }
                    for spec_hash in case.all_specs
                ],
                "default_tokens": (
                    list(_PREFILL_TOKEN_DEFAULTS) if case.prefill else [1]
                ),
                "environment": dict(case.environment),
            }
            for name, case in _CASES.items()
        },
    }


def _token_sweep(raw_values: list[str], case: CaseDefinition) -> tuple[int, ...]:
    if not raw_values:
        return _PREFILL_TOKEN_DEFAULTS if case.prefill else (1,)
    values: list[int] = []
    for raw in raw_values:
        for item in raw.split(","):
            try:
                value = int(item.strip())
            except ValueError as error:
                raise ValueError(f"invalid --tokens value {item!r}") from error
            if value <= 0:
                raise ValueError(f"token sizes must be positive, got {value}")
            if value not in values:
                values.append(value)
    if not case.prefill and values != [1]:
        raise ValueError("decode exception objects must be replayed with --tokens 1")
    if case.prefill and any(value >= 4096 for value in values):
        raise ValueError(
            "these exact TF32 prefill objects use chunk_geometry=false; "
            "all requested token sizes must be below 4096"
        )
    return tuple(values)


@contextmanager
def _environment(values: tuple[tuple[str, str], ...]) -> Iterator[None]:
    previous = {name: os.environ.get(name) for name, _ in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _package_versions() -> dict[str, str]:
    result = {}
    for package in (
        "nvidia-cutlass-dsl",
        "nvidia-cutlass-dsl-libs-base",
        "nvidia-cutlass-dsl-libs-core",
        "nvidia-cutlass-dsl-libs-cu12",
        "nvidia-cutlass-dsl-libs-cu13",
    ):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "missing"
    return result


def _cpu_randn(
    shape: tuple[int, ...],
    *,
    generator: torch.Generator,
    divisor: float,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    return (
        torch.randn(shape, generator=generator, dtype=torch.float32)
        .div_(divisor)
        .to(device=device, dtype=dtype)
        .contiguous()
    )


def _post_reference(
    x: torch.Tensor,
    residual: torch.Tensor,
    prev_post: torch.Tensor,
    prev_comb: torch.Tensor,
) -> torch.Tensor:
    return (
        prev_post.unsqueeze(-1) * x.unsqueeze(1).float()
        + (prev_comb.unsqueeze(-1) * residual.unsqueeze(2).float()).sum(dim=1)
    ).to(torch.bfloat16)


def _pre_reference(
    residual: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
    norm_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flat = residual.flatten(1).float()
    mixes = F.linear(flat, fn) * torch.rsqrt(
        flat.square().mean(dim=-1, keepdim=True) + 1.0e-6
    )
    pre = torch.sigmoid(mixes[:, :4] * scale[0] + bias[:4]) + 1.0e-6
    post = 2 * torch.sigmoid(mixes[:, 4:8] * scale[1] + bias[4:8])
    comb = mixes[:, 8:].view(-1, 4, 4) * scale[2] + bias[8:].view(4, 4)
    comb = torch.softmax(comb, dim=-1) + 1.0e-6
    comb = comb / (comb.sum(dim=-2, keepdim=True) + 1.0e-6)
    for _ in range(19):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + 1.0e-6)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + 1.0e-6)
    y_raw = (pre.unsqueeze(-1) * residual.float()).sum(dim=1)
    y = (
        y_raw.to(torch.bfloat16).float()
        * torch.rsqrt(y_raw.square().mean(dim=-1, keepdim=True) + 1.0e-6)
        * norm_weight.float()
    ).to(torch.bfloat16)
    return y, post, comb


def _build_runtime(
    case: CaseDefinition,
    *,
    tokens: int,
    device: torch.device,
) -> RuntimeCase:
    hidden_size = case.hidden_size
    split_k = _SPLIT_K[hidden_size]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(2026071900 + hidden_size + tokens)
    residual_scenarios = tuple(
        _cpu_randn(
            (tokens, 4, hidden_size),
            generator=generator,
            divisor=3.0 + scenario,
            dtype=torch.bfloat16,
            device=device,
        )
        for scenario in range(2)
    )
    x_scenarios = tuple(
        _cpu_randn(
            (tokens, hidden_size),
            generator=generator,
            divisor=4.0 + scenario,
            dtype=torch.bfloat16,
            device=device,
        )
        for scenario in range(2)
    )
    residual = torch.empty_like(residual_scenarios[0])
    x = torch.empty_like(x_scenarios[0])
    fn = _cpu_randn(
        (24, 4 * hidden_size),
        generator=generator,
        divisor=64.0,
        dtype=torch.float32,
        device=device,
    )
    fn_bf16 = fn.to(torch.bfloat16).contiguous()
    pre_fn = fn.view(24, 4, hidden_size).sum(dim=1).contiguous()
    scale = _cpu_randn(
        (3,),
        generator=generator,
        divisor=3.0,
        dtype=torch.float32,
        device=device,
    )
    bias = _cpu_randn(
        (24,),
        generator=generator,
        divisor=5.0,
        dtype=torch.float32,
        device=device,
    )
    norm_weight = _cpu_randn(
        (hidden_size,),
        generator=generator,
        divisor=2.0,
        dtype=torch.bfloat16,
        device=device,
    )
    prev_post = (
        0.75
        + _cpu_randn(
            (tokens, 4),
            generator=generator,
            divisor=16.0,
            dtype=torch.float32,
            device=device,
        )
    ).contiguous()
    prev_comb = torch.softmax(
        _cpu_randn(
            (tokens, 4, 4),
            generator=generator,
            divisor=2.0,
            dtype=torch.float32,
            device=device,
        ),
        dim=1,
    ).contiguous()

    expected_m = max(_PREFILL_POLICY_MIN, tokens) if case.prefill else None
    max_tokens = tokens if expected_m is None else expected_m
    plan = plan_mhc_scratch(
        B12XMHCScratchCaps(
            device=device,
            max_tokens=max_tokens,
            hidden_size=hidden_size,
            split_k=split_k,
        )
    )
    scratch = tuple(
        torch.empty(shape, dtype=dtype, device=device)
        for shape, dtype in plan.shapes_and_dtypes()
    )
    out = torch.empty((tokens, 4, hidden_size), dtype=torch.bfloat16, device=device)
    y = torch.empty((tokens, hidden_size), dtype=torch.bfloat16, device=device)
    post = torch.empty((tokens, 4), dtype=torch.float32, device=device)
    comb = torch.empty((tokens, 4, 4), dtype=torch.float32, device=device)
    binding = plan.bind(
        scratch=scratch,
        tokens=tokens,
        expected_m=expected_m,
        y=y,
        post=post,
        comb=comb,
        out=out,
    )
    outputs = (out, post, comb, y)

    def install(scenario: int) -> None:
        x.copy_(x_scenarios[scenario])
        residual.copy_(residual_scenarios[scenario])

    def expected() -> tuple[torch.Tensor, ...]:
        if case.route == "decode-pre":
            residual_ref = x.unsqueeze(1).expand(-1, 4, -1)
            oracle_fn = fn
        else:
            residual_ref = _post_reference(x, residual, prev_post, prev_comb)
            oracle_fn = (
                fn_bf16.float() if case.route.startswith("prefill-bf16-") else fn
            )
        y_ref, post_ref, comb_ref = _pre_reference(
            residual_ref, oracle_fn, scale, bias, norm_weight
        )
        return residual_ref, post_ref, comb_ref, y_ref

    def launch() -> tuple[torch.Tensor, ...]:
        if case.route == "decode-pre":
            result = b12x_mhc_pre(
                x,
                pre_fn,
                scale,
                bias,
                rms_eps=1.0e-6,
                hc_eps=1.0e-6,
                sinkhorn_iters=20,
                norm_weight=norm_weight,
                norm_eps=1.0e-6,
                binding=binding,
            )
        else:
            result = b12x_mhc_post_pre(
                x,
                residual,
                prev_post,
                prev_comb,
                fn,
                scale,
                bias,
                rms_eps=1.0e-6,
                hc_eps=1.0e-6,
                sinkhorn_iters=20,
                norm_weight=norm_weight,
                norm_eps=1.0e-6,
                fn_bf16=(fn_bf16 if case.route.startswith("prefill-bf16-") else None),
                binding=binding,
            )
        if tuple(value.data_ptr() for value in result) != tuple(
            value.data_ptr() for value in outputs
        ):
            raise AssertionError("production API replaced caller-owned outputs")
        return result

    stable_tensors = {
        "x": x,
        "residual": residual,
        "prev_post": prev_post,
        "prev_comb": prev_comb,
        "fn": fn,
        "fn_bf16": fn_bf16,
        "pre_fn": pre_fn,
        "scale": scale,
        "bias": bias,
        "norm_weight": norm_weight,
        "scratch": scratch[0],
        "partials": binding.partials,
        "out": out,
        "post": post,
        "comb": comb,
        "y": y,
    }
    read_only_tensors = {
        name: stable_tensors[name]
        for name in (
            "prev_post",
            "prev_comb",
            "fn",
            "fn_bf16",
            "pre_fn",
            "scale",
            "bias",
            "norm_weight",
        )
    }
    scenario_tensors = {
        "residual_scenario_0": residual_scenarios[0],
        "residual_scenario_1": residual_scenarios[1],
        "x_scenario_0": x_scenarios[0],
        "x_scenario_1": x_scenarios[1],
    }
    return RuntimeCase(
        definition=case,
        tokens=tokens,
        expected_m=expected_m,
        launch=launch,
        install=install,
        expected=expected,
        outputs=outputs,
        stable_tensors=stable_tensors,
        read_only_tensors=read_only_tensors,
        scenario_tensors=scenario_tensors,
        scratch_contract={
            "planner": "plan_mhc_scratch",
            "max_tokens": max_tokens,
            "live_tokens": tokens,
            "expected_m": expected_m,
            "hidden_size": hidden_size,
            "split_k": split_k,
            "scratch_nbytes": plan.layout.nbytes,
            "scratch_shapes_and_dtypes": [
                {"shape": list(shape), "dtype": str(dtype)}
                for shape, dtype in plan.shapes_and_dtypes()
            ],
        },
    )


def _stable_pointer_check(
    tensors: Mapping[str, torch.Tensor], expected: Mapping[str, int]
) -> None:
    observed = {name: tensor.data_ptr() for name, tensor in tensors.items()}
    if observed != expected:
        raise AssertionError(f"stable tensor pointer changed: {expected} -> {observed}")


def _metrics(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    rtol: float,
    atol: float,
) -> dict[str, object]:
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    actual_f32 = actual.float()
    expected_f32 = expected.float()
    delta = actual_f32 - expected_f32
    finite = bool(torch.isfinite(actual_f32).all())
    nonzero = int(torch.count_nonzero(actual_f32))
    if not finite or nonzero == 0:
        raise AssertionError(
            f"invalid residual output: finite={finite}, nonzero={nonzero}"
        )
    denominator = torch.linalg.vector_norm(actual_f32) * torch.linalg.vector_norm(
        expected_f32
    )
    cosine = torch.sum(actual_f32 * expected_f32) / denominator.clamp_min(
        torch.finfo(torch.float32).tiny
    )
    return {
        "shape": list(actual.shape),
        "dtype": str(actual.dtype),
        "finite": finite,
        "nonzero": nonzero,
        "max_abs": float(delta.abs().max()),
        "rmse": float(torch.sqrt(torch.mean(delta.square()))),
        "cosine": float(cosine),
        "sha256": tensor_sha256(actual),
    }


def _capture_arm(
    *,
    runtime: RuntimeCase,
    compiled: Mapping[str, object],
    stream: torch.cuda.Stream,
) -> tuple[torch.cuda.CUDAGraph, list[str]]:
    observed: list[str] = []
    with (
        _environment(runtime.definition.environment),
        pin_module_launches(
            residual_kernels,
            compiled,
            observed,
        ),
    ):
        with torch.cuda.stream(stream):
            runtime.launch()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph, stream=stream):
            runtime.launch()
        stream.synchronize()
    expected_counts = Counter(runtime.definition.all_specs)
    observed_counts = Counter(observed)
    if observed_counts != Counter(
        {spec_hash: count * 2 for spec_hash, count in expected_counts.items()}
    ):
        raise RuntimeError(
            "production composite did not launch its exact object contract twice "
            f"(eager + capture): observed={observed_counts}, expected={expected_counts}"
        )
    return graph, observed


def _validate_graphs(
    *,
    runtime: RuntimeCase,
    graphs: Mapping[str, torch.cuda.CUDAGraph],
    labels: tuple[str, str],
    stream: torch.cuda.Stream,
    stage: str,
    scenario: int,
    stable_pointers: Mapping[str, int],
) -> dict[str, object]:
    with torch.cuda.stream(stream):
        runtime.install(scenario)
    stream.synchronize()
    expected = runtime.expected()
    if all(torch.equal(expected[0], value) for value in expected[1:]):
        raise AssertionError("residual oracle outputs are unexpectedly identical")
    tolerances = (
        (0.0, 2.0e-2),
        (2.0e-4, 2.0e-4),
        (2.0e-4, 2.0e-4),
        (2.0e-2, 2.0e-2),
    )
    names = ("residual", "post", "comb", "y")
    result: dict[str, object] = {"scenario": scenario, "arms": {}}
    first_snapshot: dict[str, torch.Tensor] | None = None
    for label in labels:
        with torch.cuda.stream(stream):
            for output in runtime.outputs:
                output.fill_(float("nan"))
        stream.synchronize()
        allocator_before = allocator_counters()
        with torch.cuda.stream(stream):
            graphs[label].replay()
        stream.synchronize()
        allocator_after = allocator_counters()
        if allocator_after != allocator_before:
            raise AssertionError(
                f"{label}: graph replay allocated: {allocator_before} -> {allocator_after}"
            )
        _stable_pointer_check(runtime.stable_tensors, stable_pointers)
        arm_metrics = {
            name: _metrics(actual, reference, rtol=rtol, atol=atol)
            for name, actual, reference, (rtol, atol) in zip(
                names, runtime.outputs, expected, tolerances, strict=True
            )
        }
        snapshot = {
            name: output.clone()
            for name, output in zip(names, runtime.outputs, strict=True)
        }
        if first_snapshot is None:
            first_snapshot = snapshot
        else:
            for name in names:
                if not torch.equal(first_snapshot[name], snapshot[name]):
                    raise AssertionError(
                        f"{stage}: A/B {name} outputs are not bit exact"
                    )
        result["arms"][label] = {
            "outputs": arm_metrics,
            "allocator_before": allocator_before,
            "allocator_after": allocator_after,
            "zero_replay_allocations": True,
        }
    result.update({"arms_bit_exact": True, "passed": True})
    return result


def _run_shape(
    *,
    args: argparse.Namespace,
    case: CaseDefinition,
    tokens: int,
    labels: tuple[str, str],
    compiled: Mapping[str, Mapping[str, object]],
    gpu: Mapping[str, object],
) -> dict[str, object]:
    device = torch.device("cuda", torch.cuda.current_device())
    runtime = _build_runtime(case, tokens=tokens, device=device)
    runtime.install(0)
    stream = torch.cuda.Stream(device=device)
    stable_pointers = {
        name: tensor.data_ptr() for name, tensor in runtime.stable_tensors.items()
    }
    read_only_before = {
        name: tensor_sha256(tensor)
        for name, tensor in {
            **runtime.read_only_tensors,
            **runtime.scenario_tensors,
        }.items()
    }
    if read_only_before["x_scenario_0"] == read_only_before["x_scenario_1"]:
        raise AssertionError("x scenarios are not distinct")
    if (
        case.route != "decode-pre"
        and read_only_before["residual_scenario_0"]
        == read_only_before["residual_scenario_1"]
    ):
        raise AssertionError("residual scenarios are not distinct")

    graphs: dict[str, torch.cuda.CUDAGraph] = {}
    observed_specs: dict[str, list[str]] = {}
    for label in labels:
        graph, observed = _capture_arm(
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
    for label, topology in topologies.items():
        if int(topology["kernel_node_count"]) != len(case.all_specs):
            raise AssertionError(
                f"{label}: expected {len(case.all_specs)} graph kernel nodes, "
                f"got {topology['kernel_node_count']}"
            )

    correctness_pre = _validate_graphs(
        runtime=runtime,
        graphs=graphs,
        labels=labels,
        stream=stream,
        stage="pre_timing",
        scenario=0,
        stable_pointers=stable_pointers,
    )
    with torch.cuda.stream(stream):
        runtime.install(0)
    stream.synchronize()
    if args.cold_l2:
        make_l2_flush_fn(True, args.l2_flush_bytes)
    expected_physical_gpu = int(gpu["physical_index"])
    gpu_mode_before = gpu_mode_snapshot(expected_physical_gpu)
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
    condition_spec_hashes = list(case.all_specs)
    for condition in conditions.values():
        condition["compile_spec_hashes"] = condition_spec_hashes
        condition["all_graph_spec_hashes"] = condition_spec_hashes
    allocation_after = allocator_counters()
    gpu_mode_after = gpu_mode_snapshot(expected_physical_gpu)
    if allocation_before != allocation_after:
        raise AssertionError(
            "CUDA allocator state changed across timing: "
            f"{allocation_before} -> {allocation_after}"
        )
    correctness_post = _validate_graphs(
        runtime=runtime,
        graphs=graphs,
        labels=labels,
        stream=stream,
        stage="post_timing_live_inputs",
        scenario=1,
        stable_pointers=stable_pointers,
    )
    pre_output_hashes = {
        name: metrics["sha256"]
        for name, metrics in correctness_pre["arms"][labels[0]]["outputs"].items()
    }
    post_output_hashes = {
        name: metrics["sha256"]
        for name, metrics in correctness_post["arms"][labels[0]]["outputs"].items()
    }
    if pre_output_hashes == post_output_hashes:
        raise AssertionError("live-input mutation did not change any residual output")
    read_only_after = {
        name: tensor_sha256(tensor)
        for name, tensor in {
            **runtime.read_only_tensors,
            **runtime.scenario_tensors,
        }.items()
    }
    if read_only_after != read_only_before:
        raise AssertionError("read-only residual inputs changed during replay")
    _stable_pointer_check(runtime.stable_tensors, stable_pointers)
    return {
        "tokens": tokens,
        "hidden_size": case.hidden_size,
        "route": case.route,
        "expected_m": runtime.expected_m,
        "gpu": dict(gpu),
        "gpu_mode_before_timing": gpu_mode_before,
        "gpu_mode_after_timing": gpu_mode_after,
        "scratch_contract": runtime.scratch_contract,
        "fixed_workspace_capacity": True,
        "fixed_allocation": True,
        "same_address_arms": True,
        "fixed_pointers": stable_pointers,
        "observed_compile_specs": observed_specs,
        "cuda_graph_topology": topologies,
        "cuda_graph_topology_equal": topology_equal,
        "correctness": {
            "pre_timing": correctness_pre,
            "post_timing_live_inputs": correctness_post,
        },
        "poisoned_outputs_overwritten": True,
        "live_input_scenarios_distinct": True,
        "live_input_mutation_changed_output": True,
        "read_only_hashes": read_only_before,
        "read_only_inputs_immutable": True,
        "allocation_before_timing": allocation_before,
        "allocation_after_timing": allocation_after,
        "zero_replay_allocations": True,
        "precondition_seconds": args.precondition_seconds,
        "maximum_precondition_seconds": args.maximum_precondition_seconds,
        "max_sm_clock_delta_mhz": args.max_sm_clock_delta_mhz,
        "conditions": conditions,
    }


@torch.inference_mode()
def _run(args: argparse.Namespace) -> dict[str, object]:
    gpu = require_target_gpu()
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
    case = _CASES[args.case]
    tokens = _token_sweep(args.tokens, case)
    caches = {labels[0]: args.a_cache.resolve(), labels[1]: args.b_cache.resolve()}
    compiled: dict[str, dict[str, object]] = {label: {} for label in labels}
    artifacts: dict[str, dict[str, dict[str, Any]]] = {label: {} for label in labels}
    for label in labels:
        for spec_hash in case.all_specs:
            exact, provenance = load_exact(caches[label], spec_hash)
            if provenance["kernel_id"] != _KERNEL_ID_BY_SPEC[spec_hash]:
                raise RuntimeError(
                    f"{label}: {spec_hash} kernel id is {provenance['kernel_id']!r}, "
                    f"expected {_KERNEL_ID_BY_SPEC[spec_hash]!r}"
                )
            compiled[label][spec_hash] = exact
            artifacts[label][spec_hash] = provenance
    artifact_verification_before = {
        label: {
            spec_hash: verify_artifact(provenance)
            for spec_hash, provenance in artifacts[label].items()
        }
        for label in labels
    }
    for spec_hash in case.all_specs:
        a = artifacts[labels[0]][spec_hash]
        b = artifacts[labels[1]][spec_hash]
        if a["compile_spec_json"] != b["compile_spec_json"]:
            raise RuntimeError(f"A/B compile specifications differ for {spec_hash}")
        if a["kernel_id"] != b["kernel_id"]:
            raise RuntimeError(f"A/B kernel ids differ for {spec_hash}")

    shapes = [
        _run_shape(
            args=args,
            case=case,
            tokens=value,
            labels=labels,
            compiled=compiled,
            gpu=gpu,
        )
        for value in tokens
    ]
    artifact_verification_after = {
        label: {
            spec_hash: verify_artifact(provenance)
            for spec_hash, provenance in artifacts[label].items()
        }
        for label in labels
    }
    if artifact_verification_after != artifact_verification_before:
        raise RuntimeError("exact cache artifacts changed during benchmark")
    return {
        "schema": "b12x.residual.composite_exact_cache_abba.v1",
        "evidence_status": args.evidence_status,
        "case": {
            "name": case.name,
            "hidden_size": case.hidden_size,
            "route": case.route,
            "production_entrypoint": (
                "b12x_mhc_pre" if case.route == "decode-pre" else "b12x_mhc_post_pre"
            ),
            "target_exception_rows": [
                {
                    "family": _ROWS[spec_hash].family,
                    "compile_spec_hash": spec_hash,
                }
                for spec_hash in case.target_specs
            ],
            "all_graph_spec_hashes": list(case.all_specs),
            "environment": dict(case.environment),
        },
        "labels": {"a": labels[0], "b": labels[1]},
        "gpu": gpu,
        "provenance": {
            "command": [str(Path(sys.executable).resolve()), *sys.argv],
            "cwd": os.getcwd(),
            "git_commit": _git_output("rev-parse", "HEAD"),
            "git_worktree": _git_output("rev-parse", "--show-toplevel"),
            "git_status_short": _git_output("status", "--short").splitlines(),
            "source_sha256": {
                "benchmark": sha256_file(Path(__file__).resolve()),
                "coverage_matrix": sha256_file(
                    DATA_ROOT / "cute_migration_corpus_matrix.json"
                ),
                "shared_abba": sha256_file(CORE_ROOT / "exact_cache_abba.py"),
                "residual_api": sha256_file(REPO_ROOT / "b12x/integration/residual.py"),
                "residual_kernels": sha256_file(
                    REPO_ROOT / "b12x/integration/residual_kernels.py"
                ),
                "compiler": sha256_file(REPO_ROOT / "b12x/cute/compiler.py"),
            },
            "packages": _package_versions(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
        "artifacts": artifacts,
        "artifact_verification_before": artifact_verification_before,
        "artifact_verification_after": artifact_verification_after,
        "production_binding_or_api": True,
        "exact_cache_objects": True,
        "no_recompile": True,
        "cuda_graph_replay": True,
        "ratio_direction": "B_over_A; greater_than_1_means_CUTLASS_4.6_is_slower",
        "shapes": shapes,
    }


def main() -> None:
    args = _args()
    if args.list_cases:
        print(json.dumps(_coverage_manifest(), indent=2, sort_keys=True))
        return
    missing = [
        option
        for option, value in (
            ("--case", args.case),
            ("--a-cache", args.a_cache),
            ("--b-cache", args.b_cache),
            ("--output", args.output),
        )
        if value is None
    ]
    if missing:
        raise SystemExit(f"missing required benchmark arguments: {', '.join(missing)}")
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
                "ratios_b_over_a": {
                    str(shape["tokens"]): {
                        condition: payload["timings"]["ratios_b_over_a"]
                        for condition, payload in shape["conditions"].items()
                    }
                    for shape in result["shapes"]
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
