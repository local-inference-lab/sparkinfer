#!/usr/bin/env python3
"""Same-address graph ABBA for residual prefill partial specializations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys

import torch
import torch.nn.functional as F

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
    allocator_counters,
    gpu_mode_snapshot,
    graph_topology,
    time_conditions,
    topology_signature,
)
from validation.cutlass_migration.paths import REPO_ROOT
import b12x.cute.compiler as cute_compiler
from b12x.integration import residual_kernels


_PARTIALS = 25
_MIXES = 24
_MHC_MULT = 4
_SPLIT_K_BY_HIDDEN = {4096: 64, 7168: 112}
_BLOCK_M_SPEC_HASH = "f821ebda70f4739da06417e0292e1c755106de2710b966b932b141bd4c7ca5fe"
_COMPACT_SPEC_HASH_BY_HIDDEN = {
    4096: "e4695fe84c9f8da938c967b071c57b6e6e957b4026e1c8412d13ee3ed9e9197c",
    7168: "d26e69a632fc33843b7f6da1167605b844138bad91c830792a0d934545f70e29",
}
_FINALIZE_SPEC_HASH_BY_HIDDEN = {
    4096: "e11561fdc72b927b38e76cd1c6ca4a76ce08fd1549bd41f0f4d2b79452c46f86",
    7168: "0637890579875d27652828678921f6e1738008cd1c08aac9795ac4c5931f8df1",
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_evidence_status_argument(parser)
    add_target_gpu_argument(parser)
    parser.add_argument("--a-cache", type=pathlib.Path, required=True)
    parser.add_argument("--a-spec-hash")
    parser.add_argument("--a-label", default="cutlass-4.5.2")
    parser.add_argument("--b-cache", type=pathlib.Path, required=True)
    parser.add_argument("--b-spec-hash")
    parser.add_argument("--b-label", default="cutlass-4.6.0")
    parser.add_argument(
        "--kernel-kind", choices=("block-m", "compact"), default="block-m"
    )
    parser.add_argument(
        "--hidden-size", type=int, choices=tuple(_SPLIT_K_BY_HIDDEN), default=7168
    )
    parser.add_argument("--tokens", type=int, default=33)
    parser.add_argument("--precondition", type=int, default=200)
    parser.add_argument("--precondition-seconds", type=float, default=5.0)
    parser.add_argument("--maximum-precondition-seconds", type=float, default=60.0)
    parser.add_argument("--max-sm-clock-delta-mhz", type=float, default=60.0)
    parser.add_argument("--warmup", type=int, default=40)
    parser.add_argument("--cycles", type=int, default=500)
    parser.add_argument("--event-batch-cycles", type=int, default=50)
    parser.add_argument(
        "--cold-l2", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--l2-flush-bytes", type=int, default=0)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    return parser.parse_args()


def _file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(root: pathlib.Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _manifest_for_spec(
    cache: pathlib.Path,
    spec_hash: str,
) -> tuple[pathlib.Path, dict]:
    matches = []
    for path in cache.rglob("*.json"):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if (
            manifest.get("schema") == "b12x.cute.compile_manifest.v3"
            and manifest.get("compile_spec_hash") == spec_hash
            and manifest.get("object_sha256")
        ):
            matches.append((path, manifest))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one manifest for {spec_hash} in {cache}, got {len(matches)}"
        )
    return matches[0]


def _load_exact(cache: pathlib.Path, spec_hash: str) -> tuple[object, dict]:
    manifest_path, manifest = _manifest_for_spec(cache, spec_hash)
    object_path = manifest_path.with_suffix(".o")
    if _file_sha256(object_path) != manifest["object_sha256"]:
        raise RuntimeError(f"object digest mismatch: {object_path}")
    prior = os.environ.get("B12X_CUTE_COMPILE_CACHE_DIR")
    os.environ["B12X_CUTE_COMPILE_CACHE_DIR"] = str(cache)
    try:
        compiled = cute_compiler._load_cute_compile_from_disk(manifest["cache_key"])
    finally:
        if prior is None:
            os.environ.pop("B12X_CUTE_COMPILE_CACHE_DIR", None)
        else:
            os.environ["B12X_CUTE_COMPILE_CACHE_DIR"] = prior
    if compiled is None:
        raise RuntimeError(f"failed to load exact object {object_path}")
    return compiled, {
        "cache": str(cache.resolve()),
        "cache_key": manifest["cache_key"],
        "manifest_path": str(manifest_path),
        "manifest_sha256": _file_sha256(manifest_path),
        "compile_spec_hash": manifest["compile_spec_hash"],
        "compile_spec_json": manifest["compile_spec_json"],
        "kernel_id": manifest["kernel_id"],
        "object_path": str(object_path),
        "object_bytes": object_path.stat().st_size,
        "object_sha256": _file_sha256(object_path),
        "package_fingerprint": manifest["package_fingerprint"],
        "toolchain": manifest["toolchain"],
    }


def _verify_artifact(provenance: dict[str, object]) -> dict[str, object]:
    manifest_path = pathlib.Path(str(provenance["manifest_path"]))
    object_path = pathlib.Path(str(provenance["object_path"]))
    observed = {
        "manifest_sha256": _file_sha256(manifest_path),
        "object_sha256": _file_sha256(object_path),
        "object_bytes": object_path.stat().st_size,
    }
    if observed != {name: provenance[name] for name in observed}:
        raise RuntimeError("exact residual cache artifact changed during benchmark")
    return {"passed": True, **observed}


def _normalized_tile_spec(manifest: dict) -> tuple[dict, int]:
    spec = json.loads(manifest["compile_spec_json"])
    facts = spec.get("facts")
    if not isinstance(facts, list):
        raise RuntimeError("compile spec facts are not a list")
    tile_n = None
    normalized_facts = []
    for fact in facts:
        if isinstance(fact, list) and len(fact) == 2 and fact[0] == "tile_n":
            tile_n = int(fact[1])
            normalized_facts.append(["tile_n", "<tile_n>"])
        else:
            normalized_facts.append(fact)
    if tile_n is None:
        raise RuntimeError("compile spec has no tile_n fact")
    expected_suffix = f"_n{tile_n}"
    kernel_id = str(spec.get("kernel", ""))
    if not kernel_id.endswith(expected_suffix):
        raise RuntimeError(
            f"compile spec kernel {kernel_id!r} does not end in {expected_suffix!r}"
        )
    spec["facts"] = normalized_facts
    spec["kernel"] = re.sub(r"_n[0-9]+$", "_n<tile_n>", kernel_id)
    return spec, tile_n


def _exact_spec(manifest: dict) -> dict:
    spec = json.loads(manifest["compile_spec_json"])
    if not isinstance(spec.get("facts"), list):
        raise RuntimeError("compile spec facts are not a list")
    return spec


def _make_inputs(tokens: int, hidden_size: int) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260718)

    def randn(shape: tuple[int, ...], divisor: float) -> torch.Tensor:
        return (
            torch.randn(shape, generator=generator, dtype=torch.float32)
            .div_(divisor)
            .to("cuda")
            .contiguous()
        )

    residual = randn((tokens, _MHC_MULT, hidden_size), 3).to(torch.bfloat16)
    x = randn((tokens, hidden_size), 4).to(torch.bfloat16)
    prev_post = randn((tokens, _MHC_MULT), 3)
    prev_comb = randn((tokens, _MHC_MULT, _MHC_MULT), 4)
    fn = randn((_MIXES, _MHC_MULT * hidden_size), 64)
    scale = randn((3,), 3)
    bias = randn((_MIXES,), 5)
    norm_weight = randn((hidden_size,), 2).to(torch.bfloat16)
    return residual, x, prev_post, prev_comb, fn, scale, bias, norm_weight


def _reference(
    residual: torch.Tensor,
    x: torch.Tensor,
    prev_post: torch.Tensor,
    prev_comb: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
    norm_weight: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    residual_out = (
        prev_post.unsqueeze(-1) * x.unsqueeze(1).float()
        + (prev_comb.unsqueeze(-1) * residual.unsqueeze(2).float()).sum(dim=1)
    ).to(torch.bfloat16)
    flat = residual_out.flatten(1).float()
    mixes = F.linear(flat, fn) * torch.rsqrt(
        flat.square().mean(dim=-1, keepdim=True) + 1e-6
    )
    pre = torch.sigmoid(mixes[:, :4] * scale[0] + bias[:4]) + 1e-6
    post = 2 * torch.sigmoid(mixes[:, 4:8] * scale[1] + bias[4:8])
    comb = mixes[:, 8:].view(-1, 4, 4) * scale[2] + bias[8:].view(4, 4)
    comb = torch.softmax(comb, dim=-1) + 1e-6
    comb = comb / (comb.sum(dim=-2, keepdim=True) + 1e-6)
    for _ in range(19):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + 1e-6)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + 1e-6)
    y_fp32 = (pre.unsqueeze(-1) * residual_out.float()).sum(dim=1)
    y = (
        y_fp32.to(torch.bfloat16).float()
        * torch.rsqrt(y_fp32.square().mean(dim=-1, keepdim=True) + 1e-6)
        * norm_weight.float()
    ).to(torch.bfloat16)
    return residual_out, y, post, comb


def _error(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    difference = actual.float() - expected.float()
    return {
        "max_abs": float(difference.abs().max()),
        "rmse": float(torch.sqrt(torch.mean(difference.square()))),
    }


def _exact_comparison(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> dict[str, bool | float | int]:
    actual_fp32 = actual.float()
    expected_fp32 = expected.float()
    difference = actual_fp32 - expected_fp32
    mismatch = actual != expected
    relative = difference.abs() / expected_fp32.abs().clamp_min(
        torch.finfo(torch.float32).tiny
    )
    return {
        "exact": bool(torch.equal(actual, expected)),
        "mismatched_elements": int(mismatch.count_nonzero()),
        "total_elements": actual.numel(),
        "max_abs": float(difference.abs().max()),
        "max_relative": float(relative.max()),
        "rmse": float(torch.sqrt(torch.mean(difference.square()))),
    }


def _tensor_sha256(tensor: torch.Tensor) -> str:
    host_bytes = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(host_bytes).hexdigest()


def main() -> None:
    args = _args()
    if args.tokens <= 0:
        raise ValueError("tokens must be positive")
    if args.event_batch_cycles <= 0:
        raise ValueError("event-batch-cycles must be positive")
    if args.precondition <= 0 or args.warmup <= 0:
        raise ValueError("precondition and warmup must be positive")
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
    if args.a_label == args.b_label:
        raise ValueError("A/B labels must differ")
    if args.kernel_kind == "block-m" and args.hidden_size != 7168:
        raise ValueError(
            "the recorded block-M comparison only supports hidden-size 7168"
        )
    require_target_gpu(args.expected_physical_gpu)
    gpu_mode_initial = gpu_mode_snapshot(args.expected_physical_gpu)
    root = REPO_ROOT
    default_spec_hash = (
        _BLOCK_M_SPEC_HASH
        if args.kernel_kind == "block-m"
        else _COMPACT_SPEC_HASH_BY_HIDDEN[args.hidden_size]
    )
    a_spec_hash = args.a_spec_hash or default_spec_hash
    b_spec_hash = args.b_spec_hash or default_spec_hash
    compiled_a, manifest_a = _load_exact(args.a_cache.resolve(), a_spec_hash)
    compiled_b, manifest_b = _load_exact(args.b_cache.resolve(), b_spec_hash)
    artifact_verification_before = {
        args.a_label: _verify_artifact(manifest_a),
        args.b_label: _verify_artifact(manifest_b),
    }
    if args.kernel_kind == "block-m":
        normalized_spec_a, tile_n_a = _normalized_tile_spec(manifest_a)
        normalized_spec_b, tile_n_b = _normalized_tile_spec(manifest_b)
    else:
        normalized_spec_a = _exact_spec(manifest_a)
        normalized_spec_b = _exact_spec(manifest_b)
        tile_n_a = tile_n_b = None
    if normalized_spec_a != normalized_spec_b:
        qualifier = (
            " in facts other than tile_n" if args.kernel_kind == "block-m" else ""
        )
        raise RuntimeError(f"A/B compile specs differ{qualifier}")
    split_k = _SPLIT_K_BY_HIDDEN[args.hidden_size]
    residual, x, prev_post, prev_comb, fn, scale, bias, norm_weight = _make_inputs(
        args.tokens, args.hidden_size
    )
    partials = torch.empty(
        (args.tokens, split_k, _PARTIALS), dtype=torch.float32, device="cuda"
    )
    out = torch.empty(
        (args.tokens, _MHC_MULT, args.hidden_size),
        dtype=torch.bfloat16,
        device="cuda",
    )
    y = torch.empty(
        (args.tokens, args.hidden_size), dtype=torch.bfloat16, device="cuda"
    )
    post = torch.empty((args.tokens, _MHC_MULT), dtype=torch.float32, device="cuda")
    comb = torch.empty(
        (args.tokens, _MHC_MULT, _MHC_MULT), dtype=torch.float32, device="cuda"
    )

    def run_partial(compiled: object) -> None:
        runtime_args = (
            residual_kernels._to_kernel_tensor(
                x, residual_kernels.cutlass.BFloat16, dynamic_layout=True
            ),
            residual_kernels._to_kernel_tensor(
                residual, residual_kernels.cutlass.BFloat16, dynamic_layout=True
            ),
            residual_kernels._to_kernel_tensor(
                prev_post,
                residual_kernels.cutlass.Float32,
                assumed_align=4,
                dynamic_layout=True,
            ),
            residual_kernels._to_kernel_tensor(
                prev_comb,
                residual_kernels.cutlass.Float32,
                assumed_align=4,
                dynamic_layout=True,
            ),
            residual_kernels._to_kernel_tensor(fn, residual_kernels.cutlass.Float32),
            residual_kernels._to_kernel_tensor(
                partials,
                residual_kernels.cutlass.Float32,
                assumed_align=4,
                dynamic_layout=True,
            ),
            residual_kernels._to_kernel_tensor(
                out, residual_kernels.cutlass.BFloat16, dynamic_layout=True
            ),
            residual_kernels.Int32(args.tokens),
            residual_kernels.current_cuda_stream(),
        )
        cute_compiler.run_compiled(compiled, runtime_args)

    def run_full(compiled: object) -> None:
        run_partial(compiled)
        residual_kernels.run_mhc_finalize_gram(
            residual=out,
            partials=partials,
            scale=scale,
            bias=bias,
            y=y,
            post=post,
            comb=comb,
            rms_eps=1e-6,
            hc_eps=1e-6,
            sinkhorn_iters=20,
            norm_weight=norm_weight,
            norm_eps=1e-6,
            compact_partials=True,
            compact_projection_splits=1,
        )

    # Warm the exact objects and the common production finalize specialization.
    run_full(compiled_a)
    run_full(compiled_b)
    torch.cuda.synchronize()
    graphs: dict[str, dict[str, torch.cuda.CUDAGraph]] = {"partial": {}, "full": {}}
    for label, compiled in ((args.a_label, compiled_a), (args.b_label, compiled_b)):
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph):
            run_partial(compiled)
        graphs["partial"][label] = graph
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph):
            run_full(compiled)
        graphs["full"][label] = graph
    torch.cuda.synchronize()
    graph_topologies = {
        mode: {label: graph_topology(graph) for label, graph in mode_graphs.items()}
        for mode, mode_graphs in graphs.items()
    }
    graph_topologies_equal = all(
        topology_signature(topologies[args.a_label])
        == topology_signature(topologies[args.b_label])
        for topologies in graph_topologies.values()
    )
    if not graph_topologies_equal:
        raise AssertionError("A/B residual CUDA graph topologies differ")

    expected_scenario_0 = _reference(
        residual, x, prev_post, prev_comb, fn, scale, bias, norm_weight
    )
    output_names = ("residual", "y", "post", "comb")
    oracle_tolerances = (
        {"rtol": 0.0, "atol": 2e-2},
        {"rtol": 2e-2, "atol": 2e-2},
        {"rtol": 2e-4, "atol": 2e-4},
        {"rtol": 2e-4, "atol": 2e-4},
    )

    def validate_arms(
        expected: tuple[torch.Tensor, ...],
        *,
        scenario: int,
    ) -> tuple[dict[str, object], dict[str, tuple[torch.Tensor, ...]]]:
        arm_outputs: dict[str, tuple[torch.Tensor, ...]] = {}
        allocator_checks = {}
        for label in (args.a_label, args.b_label):
            partials.fill_(float("nan"))
            out.fill_(float("nan"))
            y.fill_(float("nan"))
            post.fill_(float("nan"))
            comb.fill_(float("nan"))
            allocator_before = allocator_counters()
            graphs["full"][label].replay()
            torch.cuda.synchronize()
            allocator_after = allocator_counters()
            if allocator_after != allocator_before:
                raise AssertionError(
                    f"{label} changed allocator state during scenario {scenario} replay"
                )
            allocator_checks[label] = {
                "before": allocator_before,
                "after": allocator_after,
            }
            actual = (out.clone(), y.clone(), post.clone(), comb.clone())
            if any(not bool(torch.isfinite(value).all()) for value in actual):
                raise AssertionError(f"{label} left poisoned residual outputs")
            arm_outputs[label] = actual
            for value, reference, tolerance in zip(
                actual, expected, oracle_tolerances, strict=True
            ):
                torch.testing.assert_close(value, reference, **tolerance)
        correctness = {
            label: {
                name: _error(value, reference)
                for name, value, reference in zip(
                    output_names, values, expected, strict=True
                )
            }
            for label, values in arm_outputs.items()
        }
        arm_comparison = {
            name: _exact_comparison(a_value, b_value)
            for name, a_value, b_value in zip(
                output_names,
                arm_outputs[args.a_label],
                arm_outputs[args.b_label],
                strict=True,
            )
        }
        exact_arm_equality = all(
            bool(comparison["exact"]) for comparison in arm_comparison.values()
        )
        if not exact_arm_equality:
            raise AssertionError("A/B residual outputs are not bit exact")
        return {
            "scenario": scenario,
            "correctness": correctness,
            "arm_comparison": arm_comparison,
            "arm_output_sha256": {
                label: {
                    name: _tensor_sha256(value)
                    for name, value in zip(output_names, values, strict=True)
                }
                for label, values in arm_outputs.items()
            },
            "exact_arm_equality": exact_arm_equality,
            "allocator_checks": allocator_checks,
        }, arm_outputs

    fixed_tensors = {
        "residual": residual,
        "x": x,
        "prev_post": prev_post,
        "prev_comb": prev_comb,
        "fn": fn,
        "scale": scale,
        "bias": bias,
        "norm_weight": norm_weight,
        "partials": partials,
        "out": out,
        "y": y,
        "post": post,
        "comb": comb,
    }
    fixed_pointers = {name: tensor.data_ptr() for name, tensor in fixed_tensors.items()}
    read_only_tensors = {
        "residual": residual,
        "prev_post": prev_post,
        "prev_comb": prev_comb,
        "fn": fn,
        "scale": scale,
        "bias": bias,
        "norm_weight": norm_weight,
    }
    read_only_before = {
        name: _tensor_sha256(tensor) for name, tensor in read_only_tensors.items()
    }
    scenario_0_input_hash = _tensor_sha256(x)
    pre_timing_validation, scenario_0_outputs_pre = validate_arms(
        expected_scenario_0, scenario=0
    )
    # Materialize the fixed cold-L2 sweep before the replay-allocation baseline.
    if args.cold_l2:
        make_l2_flush_fn(True, args.l2_flush_bytes)
    stream = torch.cuda.current_stream()
    gpu_mode_before_timing = gpu_mode_snapshot(args.expected_physical_gpu)
    allocation_before_timing = allocator_counters()
    conditions_by_mode = {
        mode: time_conditions(
            mode_graphs,
            labels=(args.a_label, args.b_label),
            precondition=args.precondition,
            warmup=args.warmup,
            cycles=args.cycles,
            event_batch_cycles=args.event_batch_cycles,
            stream=stream,
            cold_l2=args.cold_l2,
            l2_flush_bytes=args.l2_flush_bytes,
            precondition_seconds=args.precondition_seconds,
            maximum_precondition_seconds=args.maximum_precondition_seconds,
            mode_snapshot=lambda: gpu_mode_snapshot(args.expected_physical_gpu),
            required_pstate="P1",
            max_sm_clock_delta_mhz=args.max_sm_clock_delta_mhz,
        )
        for mode, mode_graphs in graphs.items()
    }
    compile_spec_hashes = {
        args.a_label: a_spec_hash,
        args.b_label: b_spec_hash,
    }
    target_spec_hashes = sorted(set(compile_spec_hashes.values()))
    all_graph_spec_hashes_by_mode = {
        "partial": target_spec_hashes,
        "full": target_spec_hashes,
    }
    non_comparison_graph_spec_hashes_by_mode = {
        "partial": [],
        "full": [_FINALIZE_SPEC_HASH_BY_HIDDEN[args.hidden_size]],
    }
    for mode, conditions in conditions_by_mode.items():
        for condition in conditions.values():
            condition["compile_spec_hashes"] = compile_spec_hashes
            condition["all_graph_spec_hashes"] = all_graph_spec_hashes_by_mode[mode]
            condition["non_comparison_graph_spec_hashes"] = (
                non_comparison_graph_spec_hashes_by_mode[mode]
            )
    allocation_after_timing = allocator_counters()
    gpu_mode_after_timing = gpu_mode_snapshot(args.expected_physical_gpu)
    if allocation_after_timing != allocation_before_timing:
        raise AssertionError("CUDA allocator state changed during residual timing")
    post_timing_pointers = {
        name: tensor.data_ptr() for name, tensor in fixed_tensors.items()
    }
    if post_timing_pointers != fixed_pointers:
        raise RuntimeError("tensor addresses changed during graph timing")
    scenario_0_after_timing = _tensor_sha256(x)
    if scenario_0_after_timing != scenario_0_input_hash:
        raise AssertionError("timed residual live input changed")
    post_timing_validation, scenario_0_outputs = validate_arms(
        expected_scenario_0, scenario=0
    )
    if any(
        not torch.equal(
            scenario_0_outputs_pre[label][index], scenario_0_outputs[label][index]
        )
        for label in (args.a_label, args.b_label)
        for index in range(len(output_names))
    ):
        raise AssertionError("residual scenario-0 output changed across timing")

    x.mul_(-0.5).add_(0.0625)
    scenario_1_input_hash = _tensor_sha256(x)
    if scenario_1_input_hash == scenario_0_after_timing:
        raise AssertionError("residual live-input mutation did not change x")
    expected_scenario_1 = _reference(
        residual, x, prev_post, prev_comb, fn, scale, bias, norm_weight
    )
    live_validation, scenario_1_outputs = validate_arms(expected_scenario_1, scenario=1)
    changed_outputs = {
        label: any(
            not torch.equal(
                scenario_0_outputs[label][index], scenario_1_outputs[label][index]
            )
            for index in range(len(output_names))
        )
        for label in (args.a_label, args.b_label)
    }
    if not all(changed_outputs.values()):
        raise AssertionError(
            f"residual live mutation did not change every arm: {changed_outputs}"
        )
    read_only_after = {
        name: _tensor_sha256(tensor) for name, tensor in read_only_tensors.items()
    }
    if read_only_after != read_only_before:
        raise AssertionError("read-only residual inputs changed during benchmark")
    post_live_pointers = {
        name: tensor.data_ptr() for name, tensor in fixed_tensors.items()
    }
    if post_live_pointers != fixed_pointers:
        raise RuntimeError("tensor addresses changed during residual live mutation")
    artifact_verification_after = {
        args.a_label: _verify_artifact(manifest_a),
        args.b_label: _verify_artifact(manifest_b),
    }
    if artifact_verification_after != artifact_verification_before:
        raise RuntimeError("exact residual cache artifacts changed during benchmark")
    gpu_mode_final = gpu_mode_snapshot(args.expected_physical_gpu)
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    result = {
        "schema": "b12x.residual_prefill_partial.cache_abba.v4",
        "evidence_status": args.evidence_status,
        "provenance": {
            "command": [str(pathlib.Path(sys.executable).resolve()), *sys.argv],
            "git_commit": _git_output(root, "rev-parse", "HEAD"),
            "git_worktree": _git_output(root, "rev-parse", "--show-toplevel"),
            "git_status_short": _git_output(root, "status", "--short").splitlines(),
            "source_sha256": {
                "benchmark": _file_sha256(pathlib.Path(__file__).resolve()),
                "residual_kernels": _file_sha256(
                    root / "b12x/integration/residual_kernels.py"
                ),
                "compiler": _file_sha256(root / "b12x/cute/compiler.py"),
            },
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
        "shape": {
            "tokens": args.tokens,
            "hidden_size": args.hidden_size,
            "split_k": split_k,
            "kernel_kind": args.kernel_kind,
            "block_m": 2 if args.kernel_kind == "block-m" else 1,
            "tile_n_by_label": {
                args.a_label: tile_n_a,
                args.b_label: tile_n_b,
            },
        },
        "labels": {"a": args.a_label, "b": args.b_label},
        "manifests": {"a": manifest_a, "b": manifest_b},
        "artifact_verification_before": artifact_verification_before,
        "artifact_verification_after": artifact_verification_after,
        "gpu": {
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "physical_index": args.expected_physical_gpu,
            "logical_device": torch.cuda.current_device(),
            "name": props.name,
            "uuid": str(getattr(props, "uuid", "")),
            "capability": list(torch.cuda.get_device_capability()),
        },
        "gpu_mode_initial": gpu_mode_initial,
        "gpu_mode_final": gpu_mode_final,
        "gpu_mode_before_timing": gpu_mode_before_timing,
        "gpu_mode_after_timing": gpu_mode_after_timing,
        "graph_replay": True,
        "cuda_graph_topology": graph_topologies,
        "cuda_graph_topology_equal": graph_topologies_equal,
        "fixed_allocation": True,
        "fixed_workspace_capacity": True,
        "same_address_arms": True,
        "fixed_pointers": fixed_pointers,
        "post_timing_pointers": post_timing_pointers,
        "post_live_mutation_pointers": post_live_pointers,
        "oracle_tolerances": dict(zip(output_names, oracle_tolerances, strict=True)),
        "pre_timing_validation": pre_timing_validation,
        "post_timing_validation": post_timing_validation,
        "correctness": {
            "scenario_0_pre_timing": pre_timing_validation,
            "scenario_0_post_timing": post_timing_validation,
            "scenario_1_live_mutation": live_validation,
        },
        "arms_bit_exact": True,
        "poisoned_outputs_overwritten": True,
        "read_only_inputs_unchanged": True,
        "read_only_input_sha256": read_only_before,
        "read_only_inputs": {
            "unchanged": True,
            "sha256_before": read_only_before,
            "sha256_after": read_only_after,
            "timed_live_scenario_0": {
                "unchanged": True,
                "sha256_before": {"x": scenario_0_input_hash},
                "sha256_after": {"x": scenario_0_after_timing},
            },
        },
        "live_input_scenarios_distinct": True,
        "live_input_mutation_changed_input": True,
        "live_input_mutation_changed_output": True,
        "live_input_mutation": {
            "captured_graph_reused": True,
            "in_place": True,
            "same_addresses": True,
            "scenarios_distinct": True,
            "changed_input": True,
            "changed_inputs": {"x": True},
            "changed_output": True,
            "changed_outputs": changed_outputs,
            "scenario_0": {"x_sha256": scenario_0_input_hash},
            "scenario_1": {"x_sha256": scenario_1_input_hash},
            "scenario_0_output_sha256": post_timing_validation["arm_output_sha256"],
            "scenario_1_output_sha256": live_validation["arm_output_sha256"],
            "allocation_addresses_stable": True,
        },
        "allocator_stable": True,
        "zero_replay_allocations": True,
        "allocation_before_timing": allocation_before_timing,
        "allocation_after_timing": allocation_after_timing,
        "cold_l2": args.cold_l2,
        "l2_flush_bytes": resolve_l2_flush_bytes(args.l2_flush_bytes),
        "precondition": args.precondition,
        "precondition_seconds": args.precondition_seconds,
        "maximum_precondition_seconds": args.maximum_precondition_seconds,
        "max_sm_clock_delta_mhz": args.max_sm_clock_delta_mhz,
        "warmup_cycles": args.warmup,
        "cycles": args.cycles,
        "event_batch_cycles": args.event_batch_cycles,
        "compile_spec_hashes": compile_spec_hashes,
        "all_graph_spec_hashes_by_mode": all_graph_spec_hashes_by_mode,
        "non_comparison_graph_spec_hashes_by_mode": (
            non_comparison_graph_spec_hashes_by_mode
        ),
        "conditions_by_mode": conditions_by_mode,
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
                "ratios_b_over_a": {
                    mode: {
                        condition: condition_result["timings"]["ratios_b_over_a"]
                        for condition, condition_result in conditions.items()
                    }
                    for mode, conditions in conditions_by_mode.items()
                },
                "summaries": {
                    mode: {
                        condition: {
                            label: {
                                metric: summary[f"{metric}_us"]
                                for metric in (
                                    "mean",
                                    "trimmed_mean",
                                    "median",
                                    "min",
                                    "p95",
                                )
                            }
                            for label, summary in condition_result["timings"][
                                "summaries"
                            ].items()
                        }
                        for condition, condition_result in conditions.items()
                    }
                    for mode, conditions in conditions_by_mode.items()
                },
            },
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
