#!/usr/bin/env python3
"""Compare two cached BF16->FP4 kernel objects in one GPU process.

This is intentionally cache-object based: it keeps input, graph capture, CUDA
context, clock preconditioning, and timing order identical while selecting two
different package fingerprints and object-cache roots.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import pathlib
import subprocess
import sys

import torch

from benchmarks.common import make_l2_flush_fn
from validation.cutlass_migration.core.evidence_status import (
    add_evidence_status_argument,
)
from validation.cutlass_migration.core.gpu_scope import (
    add_target_gpu_argument,
    require_target_gpu,
)
from validation.cutlass_migration.core.exact_cache_abba import (
    allocator_counters,
    artifact_provenance,
    gpu_mode_snapshot,
    graph_topology,
    time_conditions,
    topology_signature,
    verify_artifact,
)
from validation.cutlass_migration.paths import REPO_ROOT
import sparkinfer.cute.compiler as cute_compiler
from sparkinfer.cute.intrinsics import quantize_grouped_nvfp4_torch
import sparkinfer.quantization as quantization


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_evidence_status_argument(parser)
    add_target_gpu_argument(parser)
    parser.add_argument("--a-cache", type=pathlib.Path, required=True)
    parser.add_argument("--a-fingerprint", required=True)
    parser.add_argument("--a-label", default="cutlass-4.5.2")
    parser.add_argument(
        "--a-spec-mode", choices=("current", "v1-mk"), default="current"
    )
    parser.add_argument("--b-cache", type=pathlib.Path, required=True)
    parser.add_argument("--b-fingerprint", required=True)
    parser.add_argument("--b-label", default="cutlass-4.6.0")
    parser.add_argument(
        "--b-spec-mode", choices=("current", "v1-mk"), default="current"
    )
    parser.add_argument("--M", type=int, required=True)
    parser.add_argument("--K", type=int, required=True)
    parser.add_argument("--global-scale", type=float, default=1.0)
    parser.add_argument("--precondition", type=int, default=2000)
    parser.add_argument("--precondition-seconds", type=float, default=5.0)
    parser.add_argument("--maximum-precondition-seconds", type=float, default=60.0)
    parser.add_argument("--max-sm-clock-delta-mhz", type=float, default=60.0)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--cycles", type=int, default=1000)
    parser.add_argument("--event-batch-cycles", type=int, default=100)
    parser.add_argument("--replays-per-reported-sample", type=int, default=1)
    parser.add_argument("--l2-flush-bytes", type=int, default=0)
    parser.add_argument("--no-flush-l2", action="store_true")
    parser.add_argument(
        "--separate-output",
        action="store_true",
        help="Use distinct output addresses (shared addresses are less confounded).",
    )
    parser.add_argument("--output", type=pathlib.Path, required=True)
    return parser.parse_args()


def _sha256(tensor: torch.Tensor) -> str:
    data = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_output(repo_root: pathlib.Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _provenance(repo_root: pathlib.Path) -> dict[str, object]:
    tracked_diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    ).stdout
    source_paths = (
        pathlib.Path(__file__).resolve(),
        repo_root / "sparkinfer" / "quantization" / "__init__.py",
        repo_root / "sparkinfer" / "quantization" / "bf16_to_fp4_tma.py",
        repo_root / "sparkinfer" / "cute" / "compiler.py",
        repo_root / "sparkinfer" / "cute" / "fp4.py",
    )
    package_versions = {}
    for package in ("nvidia-cutlass-dsl", "torch"):
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = None
    return {
        "command": [str(pathlib.Path(sys.executable).resolve()), *sys.argv],
        "cwd": os.getcwd(),
        "git_commit": _git_output(repo_root, "rev-parse", "HEAD"),
        "git_worktree": _git_output(repo_root, "rev-parse", "--show-toplevel"),
        "git_status_short": _git_output(repo_root, "status", "--short").splitlines(),
        "git_tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
        "source_sha256": {
            str(path.relative_to(repo_root)): _file_sha256(path)
            for path in source_paths
        },
        "package_versions": package_versions,
        "torch_cuda_version": torch.version.cuda,
    }


def _load(
    cache: pathlib.Path,
    fingerprint: str,
    M: int,
    K: int,
    spec_mode: str,
):
    os.environ["SPARKINFER_CUTE_COMPILE_CACHE_DIR"] = str(cache)
    cute_compiler._sparkinfer_package_fingerprint = lambda: fingerprint
    cute_compiler.clear_compile_cache()
    quantization._KERNEL_CACHE.clear()
    original_spec_factory = quantization.KernelCompileSpec
    original_disk_load = cute_compiler._load_cute_compile_from_disk
    observed_cache_keys: list[str] = []

    def tracked_disk_load(cache_key: str):
        observed_cache_keys.append(cache_key)
        return original_disk_load(cache_key)

    cute_compiler._load_cute_compile_from_disk = tracked_disk_load
    if spec_mode == "v1-mk":
        # Load an immutable pre-v2 artifact for same-process A/B timing.  The
        # old explicit cache contract used exactly facts=(M,K); current source
        # state is irrelevant on an exact disk hit and no compilation is
        # permitted below.
        class _V1MKSpecFactory:
            @staticmethod
            def from_key(kernel_id, _version, _key, *, labels=None):
                del labels
                return original_spec_factory.from_key(kernel_id, 1, (M, K))

        quantization.KernelCompileSpec = _V1MKSpecFactory
    try:
        launch = quantization.compile_bf16_to_fp4_tma(M, K)
    finally:
        quantization.KernelCompileSpec = original_spec_factory
        cute_compiler._load_cute_compile_from_disk = original_disk_load
    info = cute_compiler.compile_cache_info()
    if info["compile_misses"] or info["disk_cache_hits"] != 1:
        raise RuntimeError(f"expected one exact disk-cache hit, got {info}")
    unique_keys = set(observed_cache_keys)
    if len(unique_keys) != 1:
        raise RuntimeError(
            f"expected one exact disk cache key, observed {sorted(unique_keys)}"
        )
    cache_key = unique_keys.pop()
    manifest_path = cache.resolve() / cache_key[:2] / f"{cache_key}.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    provenance = artifact_provenance(
        cache,
        str(manifest["compile_spec_hash"]),
        cache_key=cache_key,
    )
    if provenance["package_fingerprint"] != fingerprint:
        raise RuntimeError("exact BF16->FP4 artifact fingerprint mismatch")
    return launch, info, provenance


def _guarded(length: int, value: int) -> tuple[torch.Tensor, torch.Tensor]:
    guard = 4096
    storage = torch.full((length + 2 * guard,), value, dtype=torch.uint8, device="cuda")
    return storage, storage[guard : guard + length]


def main() -> None:
    args = _args()
    if args.cycles < 500:
        raise ValueError("--cycles must be at least 500 for this comparison")
    if args.cycles % 2:
        raise ValueError("--cycles must be even for balanced ABBA/BAAB timing")
    if args.precondition < 0 or args.warmup < 1:
        raise ValueError("invalid precondition/warmup settings")
    if args.event_batch_cycles < 1:
        raise ValueError("--event-batch-cycles must be positive")
    if args.replays_per_reported_sample < 1:
        raise ValueError("--replays-per-reported-sample must be positive")
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
    if args.separate_output:
        raise ValueError("release ABBA evidence requires same-address arm outputs")
    if args.no_flush_l2 and args.evidence_status == "final-source":
        raise ValueError("final-source evidence requires warm- and cold-L2 timing")
    provenance = _provenance(REPO_ROOT)
    require_target_gpu(args.expected_physical_gpu)
    gpu_mode_initial = gpu_mode_snapshot(args.expected_physical_gpu)
    # HardwareInfo queries the current driver context directly; materialize a
    # CUDA allocation first instead of relying on a metadata query to do so.
    torch.empty(1, dtype=torch.uint8, device="cuda")
    if args.M % 128 or args.K % 128:
        raise ValueError("M and K must be multiples of 128")

    original_fingerprint = cute_compiler._sparkinfer_package_fingerprint
    try:
        launch_a, cache_a, artifact_a = _load(
            args.a_cache,
            args.a_fingerprint,
            args.M,
            args.K,
            args.a_spec_mode,
        )
        launch_b, cache_b, artifact_b = _load(
            args.b_cache,
            args.b_fingerprint,
            args.M,
            args.K,
            args.b_spec_mode,
        )
    finally:
        cute_compiler._sparkinfer_package_fingerprint = original_fingerprint
    artifacts = {args.a_label: artifact_a, args.b_label: artifact_b}
    if artifact_a["compile_spec_json"] != artifact_b["compile_spec_json"]:
        raise RuntimeError("A/B compile specifications differ")
    if artifact_a["kernel_id"] != artifact_b["kernel_id"]:
        raise RuntimeError("A/B kernel IDs differ")
    artifact_verification_before = {
        label: verify_artifact(record) for label, record in artifacts.items()
    }

    generator = torch.Generator(device="cpu")
    generator.manual_seed(91700 + args.M + args.K)
    source = (
        torch.randn((args.M, args.K), generator=generator, dtype=torch.float32)
        .div_(4.0)
        .to("cuda", dtype=torch.bfloat16)
        .contiguous()
    )
    global_scale = torch.tensor([args.global_scale], dtype=torch.float32, device="cuda")
    row_counts = torch.tensor([args.M], dtype=torch.int32, device="cuda")

    def current_reference() -> tuple[torch.Tensor, torch.Tensor]:
        packed, scale_view = quantize_grouped_nvfp4_torch(
            source.unsqueeze(0), row_counts, global_scale
        )
        scale = (
            scale_view.permute(5, 2, 4, 0, 1, 3)
            .contiguous()
            .view(torch.uint8)
            .reshape(-1)
        )
        return packed, scale

    packed_ref, scale_ref = current_reference()
    packed_len = args.M * args.K // 2
    scale_len = args.M * args.K // 16
    packed_guards: dict[str, torch.Tensor] = {}
    scale_guards: dict[str, torch.Tensor] = {}
    outputs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    graphs: dict[str, torch.cuda.CUDAGraph] = {}
    launches = {args.a_label: launch_a, args.b_label: launch_b}
    shared_guarded = None
    if not args.separate_output:
        shared_guarded = (
            *_guarded(packed_len, 0xA5),
            *_guarded(scale_len, 0xC5),
        )
    for index, (label, launch) in enumerate(launches.items()):
        if shared_guarded is None:
            packed_guard, packed = _guarded(packed_len, 0xA5 + index)
            scale_guard, scale = _guarded(scale_len, 0xC5 + index)
        else:
            packed_guard, packed, scale_guard, scale = shared_guarded
        packed_guards[label] = packed_guard
        scale_guards[label] = scale_guard
        outputs[label] = (packed, scale)
        launch(source, global_scale, packed, scale)
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph):
            launch(source, global_scale, packed, scale)
        graphs[label] = graph
    torch.cuda.synchronize()

    fixed_pointers = {
        "source": source.data_ptr(),
        "global_scale": global_scale.data_ptr(),
        "outputs": {
            label: {
                "packed": packed.data_ptr(),
                "scale": scale.data_ptr(),
            }
            for label, (packed, scale) in outputs.items()
        },
    }

    scenario_0_input_hash = _sha256(source)
    read_only_initial = {
        "global_scale": _sha256(global_scale),
        "row_counts": _sha256(row_counts),
    }
    packed_ref_hash = _sha256(packed_ref)
    scale_ref_hash = _sha256(scale_ref)

    topologies = {label: graph_topology(graph) for label, graph in graphs.items()}
    topology_equal = topology_signature(topologies[args.a_label]) == topology_signature(
        topologies[args.b_label]
    )
    if not topology_equal:
        raise AssertionError("A/B CUDA graph topology differs")

    def validate(
        label: str,
        reference: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[dict[str, object], tuple[torch.Tensor, torch.Tensor]]:
        packed_ref_current, scale_ref_current = reference
        assert source.data_ptr() == fixed_pointers["source"]
        assert global_scale.data_ptr() == fixed_pointers["global_scale"]
        packed, scale = outputs[label]
        assert packed.data_ptr() == fixed_pointers["outputs"][label]["packed"]
        assert scale.data_ptr() == fixed_pointers["outputs"][label]["scale"]
        actual = packed.view(1, args.M, args.K // 2).permute(1, 2, 0)
        torch.testing.assert_close(actual, packed_ref_current, rtol=0.0, atol=0.0)
        torch.testing.assert_close(scale, scale_ref_current, rtol=0.0, atol=0.0)
        index = tuple(launches).index(label)
        guard = 4096
        packed_guard_value = 0xA5 + index if shared_guarded is None else 0xA5
        scale_guard_value = 0xC5 + index if shared_guarded is None else 0xC5
        assert torch.all(packed_guards[label][:guard] == packed_guard_value)
        assert torch.all(packed_guards[label][-guard:] == packed_guard_value)
        assert torch.all(scale_guards[label][:guard] == scale_guard_value)
        assert torch.all(scale_guards[label][-guard:] == scale_guard_value)
        metrics = {
            "passed": True,
            "packed": {
                "max_abs": float(
                    (actual.to(torch.int16) - packed_ref_current.to(torch.int16))
                    .abs()
                    .max()
                )
            },
            "scale": {
                "max_abs": float(
                    (scale.to(torch.int16) - scale_ref_current.to(torch.int16))
                    .abs()
                    .max()
                )
            },
        }
        return metrics, (actual.clone(), scale.clone())

    def replay_scenario(
        scenario: int,
        reference: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[
        dict[str, dict[str, object]], dict[str, tuple[torch.Tensor, torch.Tensor]]
    ]:
        scenario_correctness: dict[str, dict[str, object]] = {}
        arm_outputs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for label in launches:
            packed, scale = outputs[label]
            poison_replays = 0
            metrics: dict[str, object] | None = None
            snapshot: tuple[torch.Tensor, torch.Tensor] | None = None
            allocator_checks = []
            for poison in (0x5A, 0xA5):
                packed.fill_(poison)
                scale.fill_(poison)
                allocator_before = allocator_counters()
                graphs[label].replay()
                torch.cuda.synchronize()
                allocator_after = allocator_counters()
                if allocator_after != allocator_before:
                    raise AssertionError(
                        f"{label} changed allocator state during scenario "
                        f"{scenario} replay"
                    )
                allocator_checks.append(
                    {"before": allocator_before, "after": allocator_after}
                )
                metrics, snapshot = validate(label, reference)
                poison_replays += 1
            assert metrics is not None and snapshot is not None
            metrics.update(
                {
                    "scenario": scenario,
                    "poison_replays": poison_replays,
                    "allocator_checks": allocator_checks,
                }
            )
            scenario_correctness[label] = metrics
            arm_outputs[label] = snapshot
        if not all(
            torch.equal(
                arm_outputs[args.a_label][index], arm_outputs[args.b_label][index]
            )
            for index in range(2)
        ):
            raise AssertionError(
                f"scenario {scenario}: A/B BF16->FP4 outputs are not bit exact"
            )
        return scenario_correctness, arm_outputs

    correctness_pre, scenario_0_outputs_pre = replay_scenario(
        0, (packed_ref, scale_ref)
    )
    labels = (args.a_label, args.b_label)
    if not args.no_flush_l2:
        # Populate the stable flush-buffer cache before the outer allocation
        # snapshot. The shared timer binds the exact capacity to its cold-L2
        # condition and checks allocation stability internally as well.
        make_l2_flush_fn(True, args.l2_flush_bytes)
    allocation_before_timing = allocator_counters()
    conditions = time_conditions(
        graphs,
        labels=labels,
        precondition=args.precondition,
        warmup=args.warmup,
        cycles=args.cycles,
        event_batch_cycles=args.event_batch_cycles,
        stream=torch.cuda.current_stream(),
        cold_l2=not args.no_flush_l2,
        l2_flush_bytes=args.l2_flush_bytes,
        replays_per_reported_sample=args.replays_per_reported_sample,
        precondition_seconds=args.precondition_seconds,
        maximum_precondition_seconds=args.maximum_precondition_seconds,
        mode_snapshot=lambda: gpu_mode_snapshot(args.expected_physical_gpu),
        required_pstate="P1",
        max_sm_clock_delta_mhz=args.max_sm_clock_delta_mhz,
    )
    allocation_after_timing = allocator_counters()
    if allocation_after_timing != allocation_before_timing:
        raise AssertionError("CUDA allocator state changed during graph timing")
    compile_spec_hashes = {
        label: str(record["compile_spec_hash"]) for label, record in artifacts.items()
    }
    all_graph_spec_hashes = sorted(set(compile_spec_hashes.values()))
    for condition in conditions.values():
        condition["compile_spec_hashes"] = compile_spec_hashes
        condition["all_graph_spec_hashes"] = all_graph_spec_hashes
    scenario_0_after_timing = _sha256(source)
    if scenario_0_after_timing != scenario_0_input_hash:
        raise AssertionError("timed BF16 source changed before live mutation")
    correctness_post, scenario_0_outputs = replay_scenario(0, (packed_ref, scale_ref))
    if any(
        not torch.equal(
            scenario_0_outputs_pre[label][index], scenario_0_outputs[label][index]
        )
        for label in launches
        for index in range(2)
    ):
        raise AssertionError("scenario-0 output changed across BF16 timing")

    mutation_before = _sha256(source)
    source.mul_(-0.75).add_(0.03125)
    scenario_1_input_hash = _sha256(source)
    if scenario_1_input_hash == mutation_before:
        raise AssertionError("BF16 live-input mutation did not change source")
    packed_ref_live, scale_ref_live = current_reference()
    correctness_live, scenario_1_outputs = replay_scenario(
        1, (packed_ref_live, scale_ref_live)
    )
    changed_outputs = {
        label: any(
            not torch.equal(
                scenario_0_outputs[label][index], scenario_1_outputs[label][index]
            )
            for index in range(2)
        )
        for label in launches
    }
    if not all(changed_outputs.values()):
        raise AssertionError(
            f"BF16 live mutation did not change every arm output: {changed_outputs}"
        )
    read_only_final = {
        "global_scale": _sha256(global_scale),
        "row_counts": _sha256(row_counts),
    }
    if read_only_final != read_only_initial:
        raise AssertionError("read-only BF16 quantization inputs changed")
    post_correctness_pointers = {
        "source": source.data_ptr(),
        "global_scale": global_scale.data_ptr(),
        "outputs": {
            label: {"packed": packed.data_ptr(), "scale": scale.data_ptr()}
            for label, (packed, scale) in outputs.items()
        },
    }
    if post_correctness_pointers != fixed_pointers:
        raise AssertionError("BF16 tensor addresses changed across live mutation")
    artifact_verification_after = {
        label: verify_artifact(record) for label, record in artifacts.items()
    }
    if artifact_verification_after != artifact_verification_before:
        raise RuntimeError("exact BF16->FP4 cache artifacts changed during benchmark")
    gpu_mode_final = gpu_mode_snapshot(args.expected_physical_gpu)

    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    result = {
        "schema": "sparkinfer.bf16_to_fp4_tma.cache_abba.v4",
        "evidence_status": args.evidence_status,
        "provenance": provenance,
        "object_provenance": artifacts,
        "artifact_verification_before": artifact_verification_before,
        "artifact_verification_after": artifact_verification_after,
        "shape": {"M": args.M, "K": args.K},
        "global_scale": args.global_scale,
        "labels": {"a": args.a_label, "b": args.b_label},
        "cache": {
            "a": str(args.a_cache),
            "a_fingerprint": args.a_fingerprint,
            "a_spec_mode": args.a_spec_mode,
            "a_info": cache_a,
            "b": str(args.b_cache),
            "b_fingerprint": args.b_fingerprint,
            "b_spec_mode": args.b_spec_mode,
            "b_info": cache_b,
        },
        "gpu": {
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "physical_index": args.expected_physical_gpu,
            "logical_device": torch.cuda.current_device(),
            "name": props.name,
            "uuid": str(getattr(props, "uuid", "")),
            "capability": list(torch.cuda.get_device_capability()),
            "multiprocessor_count": props.multi_processor_count,
            "total_memory_bytes": props.total_memory,
        },
        "gpu_mode_initial": gpu_mode_initial,
        "gpu_mode_final": gpu_mode_final,
        "graph_replay": True,
        "cuda_graph_topology": topologies,
        "cuda_graph_topology_equal": topology_equal,
        "fixed_allocation": True,
        "fixed_workspace_capacity": True,
        "same_address_arms": True,
        "pointer_stable": True,
        "fixed_pointers": fixed_pointers,
        "shared_output_addresses": not args.separate_output,
        "orders": [
            [args.a_label, args.b_label, args.b_label, args.a_label],
            [args.b_label, args.a_label, args.a_label, args.b_label],
        ],
        "order_balance": "alternating-abba-baab",
        "precondition": args.precondition,
        "precondition_seconds": args.precondition_seconds,
        "maximum_precondition_seconds": args.maximum_precondition_seconds,
        "max_sm_clock_delta_mhz": args.max_sm_clock_delta_mhz,
        "warmup_cycles": args.warmup,
        "cycles": args.cycles,
        "event_batch_cycles": args.event_batch_cycles,
        "replays_per_reported_sample": args.replays_per_reported_sample,
        "l2_flush_bytes": args.l2_flush_bytes,
        "exact_hashes": {
            "input_scenario_0": scenario_0_input_hash,
            "input_scenario_1": scenario_1_input_hash,
            "global_scale": read_only_initial["global_scale"],
            "packed": packed_ref_hash,
            "scale": scale_ref_hash,
        },
        "canaries": True,
        "global_scale_immutable": True,
        "read_only_inputs_unchanged": True,
        "read_only_inputs": {
            "unchanged": True,
            "sha256_before": read_only_initial,
            "sha256_after": read_only_final,
            "timed_live_scenario_0": {
                "unchanged": True,
                "sha256_before": {"source": scenario_0_input_hash},
                "sha256_after": {"source": scenario_0_after_timing},
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
            "changed_inputs": {"source": True},
            "changed_output": True,
            "changed_outputs": changed_outputs,
            "scenario_0": {"source_sha256": scenario_0_input_hash},
            "scenario_1": {"source_sha256": scenario_1_input_hash},
            "scenario_0_output_sha256": {
                label: {
                    "packed": _sha256(values[0]),
                    "scale": _sha256(values[1]),
                }
                for label, values in scenario_0_outputs.items()
            },
            "scenario_1_output_sha256": {
                label: {
                    "packed": _sha256(values[0]),
                    "scale": _sha256(values[1]),
                }
                for label, values in scenario_1_outputs.items()
            },
            "allocation_addresses_stable": True,
        },
        "poisoned_outputs_overwritten": True,
        "arms_bit_exact": True,
        "correctness": {
            "scenario_0_pre_timing": correctness_pre,
            "scenario_0_post_timing": correctness_post,
            "scenario_1_live_mutation": correctness_live,
        },
        "allocator_stable": True,
        "zero_replay_allocations": True,
        "allocation_before_timing": allocation_before_timing,
        "allocation_after_timing": allocation_after_timing,
        "conditions": conditions,
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
                    name: condition["timings"]["ratios_b_over_a"]
                    for name, condition in conditions.items()
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
