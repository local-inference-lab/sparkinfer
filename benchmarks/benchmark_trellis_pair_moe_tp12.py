"""Benchmark fixed-payload P24/P33 pairs through the routed TP12 MoE path.

The benchmark uses Kimi-K3's production rank-local geometry: H=3584,
I=3072/TP12=256, top-k=16, and 896 routing slots.  It times the complete
graph-replayed route/FC1/SiTU/FC2/reduction path, not an isolated decoder.
Every P33 and P24 projection carries exactly 344,064 trellis payload bytes.

Pair cases use the public expert-static descriptor contract even when every
expert has the same kind. ``sparse`` gives a configurable small prefix of
routed experts P24 in both projections, approximating the expected production
regime after cyclic TP12 ownership. ``mixed`` alternates P24/P33 ownership
between experts and uses opposite FC1/FC2 tables as a dispatch stress case.

Example:

    B12X_COMPILE_DISK_CACHE=0 \
      python benchmarks/benchmark_trellis_pair_moe_tp12.py \
        --tokens 1,4,16 --route-fixture /path/to/layer-rank.safetensors \
        --scenarios P33,sparse --replays 200 \
        --require-known-resources --require-no-local-memory \
        --output out/trellis-pair-moe-tp12.json
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
from typing import Callable

import torch
from safetensors import safe_open

try:
    from benchmarks.common import make_l2_flush_fn, nvidia_smi_gpu_mode_snapshot
except ModuleNotFoundError:  # Direct ``python benchmarks/...py`` execution.
    from common import make_l2_flush_fn, nvidia_smi_gpu_mode_snapshot
from b12x.moe import fused_moe


_HIDDEN = 3584
_LOCAL_INTERMEDIATE = 256
_ROUTE_EXPERTS = 896
_TOPK = 16
_TILES = (64, 256, 64, 256)
_PAYLOAD_BYTES_PER_PROJECTION = _HIDDEN * _LOCAL_INTERMEDIATE * 3 // 8
_PAIR_WORDS = _PAYLOAD_BYTES_PER_PROJECTION // 2
_SCENARIOS = ("P33", "P24", "sparse", "mixed")
_FIXTURE_KIND = "kquant_qsrt_tp12_benchmark_fixture"
_FIXTURE_SCHEMA_VERSION = 1
_RESULT_KIND = "b12x_trellis_pair_moe_tp12_benchmark"
_RESULT_SCHEMA_VERSION = 1
_CODEBOOK_SOURCE_FORMATS = {"sqg_xor_cheb_t12": "qsrt_sqg_e4m3"}


@dataclass(frozen=True)
class _RouteFixture:
    path: Path
    sha256: str
    manifest: dict[str, object]
    fc1_pair_modes: torch.Tensor
    fc2_pair_modes: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor

    @property
    def num_experts(self) -> int:
        return int(self.fc1_pair_modes.numel())


@dataclass
class _RouteReplay:
    source_ids: torch.Tensor
    source_weights: torch.Tensor
    target_ids: torch.Tensor
    target_weights: torch.Tensor

    def load(self, replay: int) -> None:
        tokens = int(self.target_ids.shape[0])
        rows = int(self.source_ids.shape[0])
        if rows < tokens:
            raise ValueError(
                f"route fixture has {rows} rows, fewer than tokens={tokens}"
            )
        windows = rows - tokens + 1
        start = (replay * tokens) % windows
        self.target_ids.copy_(self.source_ids.narrow(0, start, tokens))
        self.target_weights.copy_(self.source_weights.narrow(0, start, tokens))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _load_route_fixture(path: Path) -> _RouteFixture:
    path = path.resolve()
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        try:
            manifest = json.loads(metadata["manifest"])
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("route fixture has no valid manifest") from exc
        if not isinstance(manifest, dict):
            raise ValueError("route fixture manifest must be a JSON object")
        required = {
            "fc1_pair_modes",
            "fc2_pair_modes",
            "topk_ids",
            "topk_weights",
            "observation_ids",
            "source_row_indices",
        }
        missing = sorted(required - set(handle.keys()))
        if missing:
            raise ValueError(f"route fixture is missing tensors {missing}")
        fc1 = handle.get_tensor("fc1_pair_modes")
        fc2 = handle.get_tensor("fc2_pair_modes")
        ids = handle.get_tensor("topk_ids")
        weights = handle.get_tensor("topk_weights")
        observation_ids = handle.get_tensor("observation_ids")
        source_row_indices = handle.get_tensor("source_row_indices")

    expected_manifest = {
        "kind": _FIXTURE_KIND,
        "schema_version": _FIXTURE_SCHEMA_VERSION,
        "tp_size": 12,
        "top_k": _TOPK,
        "hidden_size": _HIDDEN,
        "local_intermediate_size": _LOCAL_INTERMEDIATE,
    }
    for key, expected in expected_manifest.items():
        if manifest.get(key) != expected:
            raise ValueError(
                f"route fixture {key}={manifest.get(key)!r}, expected {expected!r}"
            )
    num_experts = int(manifest.get("num_experts", 0))
    if not 1 <= num_experts <= _ROUTE_EXPERTS:
        raise ValueError("route fixture has an invalid expert count")
    for name, modes in (("fc1", fc1), ("fc2", fc2)):
        if modes.dtype != torch.uint8 or tuple(modes.shape) != (num_experts,):
            raise ValueError(f"route fixture {name} modes have the wrong contract")
        if not bool(torch.all((modes == 0) | (modes == 1))):
            raise ValueError(f"route fixture {name} modes are not P33/P24 flags")
    if (
        ids.dtype != torch.int32
        or ids.ndim != 2
        or int(ids.shape[1]) != _TOPK
        or weights.dtype != torch.float32
        or weights.shape != ids.shape
        or not int(ids.shape[0])
    ):
        raise ValueError("route fixture top-k tensors have the wrong contract")
    if int(ids.min()) < 0 or int(ids.max()) >= num_experts:
        raise ValueError("route fixture contains an invalid expert ID")
    ordered = ids.sort(dim=1).values
    if bool(torch.any(ordered[:, 1:] == ordered[:, :-1])):
        raise ValueError("route fixture contains duplicate experts in a top-k row")
    if not bool(torch.isfinite(weights).all()) or bool(torch.any(weights < 0)):
        raise ValueError("route fixture contains invalid top-k weights")
    rows = int(ids.shape[0])
    for name, values in (
        ("observation_ids", observation_ids),
        ("source_row_indices", source_row_indices),
    ):
        if values.dtype != torch.int64 or tuple(values.shape) != (rows,):
            raise ValueError(f"route fixture {name} has the wrong contract")
    if bool(torch.any(source_row_indices < 0)) or (
        rows > 1
        and bool(torch.any(source_row_indices[1:] <= source_row_indices[:-1]))
    ):
        raise ValueError("route fixture source row indices are not strictly increasing")

    routed_fc1 = fc1[ids.to(torch.int64)]
    routed_fc2 = fc2[ids.to(torch.int64)]
    reported_rows = int(manifest.get("fixture_rows", -1))
    if reported_rows != rows:
        raise ValueError("route fixture row count disagrees with its manifest")
    reported_p24 = int(manifest.get("p24_route_executions", -1))
    if torch.equal(fc1, fc2) and reported_p24 != int(routed_fc1.sum()):
        raise ValueError("route fixture P24 route count disagrees with its manifest")
    manifest = dict(manifest)
    manifest["loaded_fc1_p24_route_fraction"] = float(routed_fc1.float().mean())
    manifest["loaded_fc2_p24_route_fraction"] = float(routed_fc2.float().mean())
    return _RouteFixture(
        path=path,
        sha256=_file_sha256(path),
        manifest=manifest,
        fc1_pair_modes=fc1,
        fc2_pair_modes=fc2,
        topk_ids=ids,
        topk_weights=weights,
    )


def _git_value(*args: str) -> str:
    try:
        return subprocess.check_output(
            ("git", *args),
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _source_sha256() -> str:
    root = Path(__file__).resolve().parents[1]
    paths = (
        Path(__file__).resolve(),
        root / "b12x/_lib/intrinsics.py",
        root / "b12x/gemm/trellis_linear/api.py",
        root / "b12x/moe/_shared/execution.py",
        root / "b12x/moe/_shared/kernels/w4a16/kernel.py",
        root / "b12x/moe/_shared/kernels/w4a16/prepare.py",
        root / "b12x/moe/fused_moe/_impl.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(root).as_posix().encode()
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


def _write_new_result(path: Path, payload: str) -> None:
    """Publish one complete benchmark result without clobbering evidence."""

    path = path.resolve()
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        descriptor = os.open(
            path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _parse_positive_list(value: str) -> list[int]:
    result = [int(item) for item in value.split(",") if item.strip()]
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("expected a non-empty positive list")
    return result


def _parse_scenarios(value: str) -> list[str]:
    aliases = {item.lower(): item for item in _SCENARIOS}
    result: list[str] = []
    for raw in value.split(","):
        key = raw.strip().lower()
        if not key:
            continue
        if key not in aliases:
            raise argparse.ArgumentTypeError(
                f"scenario must be one of {', '.join(_SCENARIOS)}, got {raw!r}"
            )
        canonical = aliases[key]
        if canonical not in result:
            result.append(canonical)
    if not result:
        raise argparse.ArgumentTypeError("at least one scenario is required")
    return result


def _mode_tables(
    scenario: str,
    experts: int,
    *,
    device: torch.device,
    sparse_p24_experts: int,
    route_fixture: _RouteFixture | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    expert_ids = torch.arange(experts, dtype=torch.int32, device=device)
    if scenario == "P33":
        fc1 = torch.zeros_like(expert_ids)
        fc2 = torch.zeros_like(expert_ids)
    elif scenario == "P24":
        fc1 = torch.ones_like(expert_ids)
        fc2 = torch.ones_like(expert_ids)
    elif scenario == "sparse":
        if route_fixture is None:
            fc1 = torch.zeros_like(expert_ids)
            fc1[:sparse_p24_experts] = 1
            fc2 = fc1.clone()
        else:
            if route_fixture.num_experts != experts:
                raise ValueError("route fixture expert count disagrees with benchmark")
            fc1 = route_fixture.fc1_pair_modes.to(device=device).to(torch.int32)
            fc2 = route_fixture.fc2_pair_modes.to(device=device).to(torch.int32)
    elif scenario == "mixed":
        fc1 = expert_ids.remainder(2).contiguous()
        fc2 = (1 - fc1).contiguous()
    else:
        raise ValueError(f"no pair tables for {scenario!r}")
    return fc1, fc2


def _routing(
    tokens: int,
    experts: int,
    *,
    device: torch.device,
    route_fixture: _RouteFixture | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if route_fixture is None:
        rows = [
            (torch.arange(_TOPK, dtype=torch.int32) + row).remainder(experts)
            for row in range(tokens)
        ]
        ids = torch.stack(rows).to(device=device).contiguous()
        weights = torch.full(
            (tokens, _TOPK),
            1.0 / _TOPK,
            dtype=torch.float32,
            device=device,
        )
    else:
        if route_fixture.num_experts != experts:
            raise ValueError("route fixture expert count disagrees with benchmark")
        if int(route_fixture.topk_ids.shape[0]) < tokens:
            raise ValueError(
                f"route fixture has fewer than tokens={tokens} route rows"
            )
        ids = route_fixture.topk_ids[:tokens].to(device=device).clone().contiguous()
        weights = (
            route_fixture.topk_weights[:tokens]
            .to(device=device)
            .clone()
            .contiguous()
        )
    expert_map = torch.full(
        (_ROUTE_EXPERTS,), -1, dtype=torch.int32, device=device
    )
    expert_map[:experts] = torch.arange(
        experts, dtype=torch.int32, device=device
    )
    return ids, weights, expert_map


def _random_i16(
    shape: tuple[int, ...],
    *,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    return torch.randint(
        -32768,
        32767,
        shape,
        dtype=torch.int16,
        device=device,
        generator=generator,
    )


def _atom_extent_from_pair_payloads(
    w13_payload: torch.Tensor,
    w2_payload: torch.Tensor,
    fc1_modes: torch.Tensor,
    fc2_modes: torch.Tensor,
    intermediate_rotations: torch.Tensor,
) -> torch.Tensor:
    """Build one canonical eight-atom extent for the kernel benchmark."""

    _, experts, pair_words = w13_payload.shape
    hidden_tiles = pair_words // (8 * 16 * 6)
    device = w13_payload.device
    atoms = torch.zeros((8, experts, 129_216), dtype=torch.uint8, device=device)

    def store_matrix(
        payload: torch.Tensor,
        modes: torch.Tensor,
        *,
        offset: int,
        fc1: bool,
    ) -> None:
        matrix_atoms = torch.zeros(
            (8, experts, 21_504), dtype=torch.int16, device=device
        )
        for mode, (low_bits, high_bits) in ((0, (3, 3)), (1, (2, 4))):
            ids = torch.nonzero(modes == mode, as_tuple=False).flatten()
            if int(ids.numel()) == 0:
                continue
            selected = payload.index_select(0, ids)
            low_words = hidden_tiles * 8 * 16 * low_bits
            if fc1:
                low = selected[:, :low_words].reshape(
                    -1, hidden_tiles, 8, 16 * low_bits
                ).permute(2, 0, 1, 3)
                high = selected[:, low_words:].reshape(
                    -1, hidden_tiles, 8, 16 * high_bits
                ).permute(2, 0, 1, 3)
            else:
                low = selected[:, :low_words].reshape(
                    -1, 8, hidden_tiles, 16 * low_bits
                ).permute(1, 0, 2, 3)
                high = selected[:, low_words:].reshape(
                    -1, 8, hidden_tiles, 16 * high_bits
                ).permute(1, 0, 2, 3)
            joined = torch.cat(
                (low.reshape(8, ids.numel(), -1), high.reshape(8, ids.numel(), -1)),
                dim=2,
            )
            selected_atoms = matrix_atoms.index_select(1, ids)
            selected_atoms[..., : joined.shape[-1]].copy_(joined)
            matrix_atoms.index_copy_(1, ids, selected_atoms)
        atoms[:, :, offset : offset + 43_008].copy_(
            matrix_atoms.contiguous().view(torch.uint8).reshape(8, experts, 43_008)
        )

    store_matrix(w13_payload[0], fc1_modes, offset=0, fc1=True)
    store_matrix(w13_payload[1], fc1_modes, offset=43_008, fc1=True)
    store_matrix(w2_payload, fc2_modes, offset=86_016, fc1=False)
    for matrix, offset in enumerate((129_024, 129_088, 129_152)):
        values = intermediate_rotations[:, matrix * 256 : (matrix + 1) * 256]
        scale_atoms = torch.cat(
            (
                values[:, :128].reshape(experts, 8, 16),
                values[:, 128:].reshape(experts, 8, 16),
            ),
            dim=2,
        ).permute(1, 0, 2).contiguous()
        atoms[:, :, offset : offset + 64].copy_(
            scale_atoms.view(torch.uint8).reshape(8, experts, 64)
        )
    return atoms


def _prepare_weights(
    scenario: str,
    experts: int,
    *,
    codebook: str,
    device: torch.device,
    generator: torch.Generator,
    sparse_p24_experts: int,
    route_fixture: _RouteFixture | None = None,
) -> fused_moe.ExpertWeights:
    weight_plan = fused_moe.plan_weights(
        quant_modes="w4a16",
        source_format=_CODEBOOK_SOURCE_FORMATS[codebook],
        activation="situ",
        params_dtype=torch.bfloat16,
        num_experts=experts,
        hidden_size=_HIDDEN,
        intermediate_size=_LOCAL_INTERMEDIATE,
        w13_layout="w13",
        trellis_bits=3,
        trellis_tile_config=_TILES,
        qsrt_storage_format="qsrt_atoms_v1",
    )
    w13 = _random_i16(
        (2, experts, _PAIR_WORDS),
        device=device,
        generator=generator,
    )
    w2 = _random_i16(
        (experts, _PAIR_WORDS),
        device=device,
        generator=generator,
    )
    fc1_modes, fc2_modes = _mode_tables(
        scenario,
        experts,
        device=device,
        sparse_p24_experts=sparse_p24_experts,
        route_fixture=route_fixture,
    )
    payload_bytes = (w13.numel() + w2.numel()) * w13.element_size()
    expected_bytes = experts * 3 * _PAYLOAD_BYTES_PER_PROJECTION
    if payload_bytes != expected_bytes:
        raise AssertionError(
            f"{scenario} payload has {payload_bytes} bytes, expected {expected_bytes}"
        )

    hidden_rotation = torch.ones(
        (1, _HIDDEN), dtype=torch.float16, device=device
    )
    intermediate_rotations = torch.ones(
        (experts, 3 * _LOCAL_INTERMEDIATE),
        dtype=torch.float16,
        device=device,
    )
    atom_payload = _atom_extent_from_pair_payloads(
        w13, w2, fc1_modes, fc2_modes, intermediate_rotations
    )
    # Use one synthetic source identity for every materialized expert so the
    # benchmark can deliberately stress homogeneous P24 as well as mixed mode
    # tables.  Production loaders supply the true distinct expert IDs.
    expert_ids = torch.zeros(experts, dtype=torch.int32, device=device)
    format_codes = (
        (fc1_modes.to(torch.uint8) << 4) | fc2_modes.to(torch.uint8)
    ).contiguous()
    return fused_moe.prepare_weights(
        plan=weight_plan,
        params_dtype=torch.bfloat16,
        qsrt_atom_payload=atom_payload,
        qsrt_first_atom_slot=0,
        qsrt_layer_index=12,
        qsrt_expert_ids=expert_ids,
        qsrt_format_codes=format_codes,
        qsrt_atom_slots=96,
        qsrt_experts_per_layer=896,
        qsrt_rotation_multiplier=5,
        gate_suh=hidden_rotation,
        up_suh=hidden_rotation,
        down_svh=hidden_rotation,
    )


def _capture_case(
    scenario: str,
    tokens: int,
    experts: int,
    *,
    codebook: str,
    device: torch.device,
    seed: int,
    warmup: int,
    sparse_p24_experts: int,
    route_fixture: _RouteFixture | None = None,
) -> tuple[
    torch.cuda.CUDAGraph,
    list[object],
    dict[str, float],
    dict[str, object],
    dict[str, object],
    _RouteReplay | None,
]:
    generator = torch.Generator(device=device).manual_seed(seed)
    weights = _prepare_weights(
        scenario,
        experts,
        codebook=codebook,
        device=device,
        generator=generator,
        sparse_p24_experts=sparse_p24_experts,
        route_fixture=route_fixture,
    )
    topk_ids, topk_weights, expert_map = _routing(
        tokens, experts, device=device, route_fixture=route_fixture
    )
    route_replay = (
        None
        if route_fixture is None
        else _RouteReplay(
            source_ids=route_fixture.topk_ids.to(device=device),
            source_weights=route_fixture.topk_weights.to(device=device),
            target_ids=topk_ids,
            target_weights=topk_weights,
        )
    )
    x = (
        torch.randn(
            (tokens, _HIDDEN),
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        * 1.0e-3
    ).to(torch.bfloat16)
    plan = fused_moe.plan(
        fused_moe.Caps(
            max_tokens=tokens,
            num_topk=_TOPK,
            route_num_experts=_ROUTE_EXPERTS,
            device=device,
            weight_plan=weights.plan,
            quant_mode="w4a16",
            w4a16_block_size_m=8,
        )
    )
    scratch_spec = plan.scratch_specs()[0]
    scratch = torch.empty(
        scratch_spec.shape,
        dtype=scratch_spec.dtype,
        device=scratch_spec.device,
    )
    binding = fused_moe.bind(
        plan,
        scratch=scratch,
        a=x,
        experts=weights,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        route_expert_map=expert_map,
        output_expert_map=expert_map,
    )
    eager = binding.run().clone()
    torch.cuda.synchronize(device)
    if not bool(torch.isfinite(eager).all()):
        raise RuntimeError(f"{scenario} produced a non-finite eager output")

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = binding.run()
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize(device)
    if not torch.equal(captured, eager):
        raise RuntimeError(f"{scenario} graph replay differs from eager output")
    correctness_summary: dict[str, object] = {
        "passed": True,
        "finite": True,
        "eager_graph_bit_exact": True,
        "independent_numeric_oracle": False,
        "reference_scope": "same compiled kernel, eager launch versus graph replay",
        "output_dtype": str(eager.dtype),
        "output_shape": list(eager.shape),
        "output_nonzero": int(torch.count_nonzero(eager)),
    }

    representation = weights.representation_for("w4a16")
    fc1_modes = representation.fc1_trellis_pair_modes
    fc2_modes = representation.fc2_trellis_pair_modes
    routed = (
        topk_ids.reshape(-1)
        if route_replay is None
        else route_replay.source_ids.reshape(-1)
    ).to(dtype=torch.long)
    mode_summary = {
        "routed_fc1_p24_fraction": (
            0.0
            if fc1_modes is None
            else float(fc1_modes.index_select(0, routed).float().mean())
        ),
        "routed_fc2_p24_fraction": (
            0.0
            if fc2_modes is None
            else float(fc2_modes.index_select(0, routed).float().mean())
        ),
    }
    fused_launch = binding.fused_launch
    registers_per_thread = int(
        getattr(fused_launch, "registers_per_thread", -1)
    )
    local_memory_bytes = int(getattr(fused_launch, "local_memory_bytes", -1))
    kernel_summary: dict[str, object] = {
        "resource_scope": "fused_route_fc1_situ_fc2",
        "kernel_symbol": getattr(fused_launch, "kernel_symbol", None),
        "resource_introspection_available": (
            registers_per_thread >= 0 and local_memory_bytes >= 0
        ),
        "registers_per_thread": registers_per_thread,
        "local_memory_bytes_per_thread": local_memory_bytes,
        "preplanned_launch_available": fused_launch is not None,
    }
    if fused_launch is not None:
        kernel_summary.update(
            {
                "cta_threads": int(fused_launch.cta_threads),
                "shared_memory_bytes": int(fused_launch.shared_memory_bytes),
                "size_m": int(fused_launch.size_m),
                "fc1_tile_n": int(fused_launch.fc1_tile_n),
                "fc1_tile_k": int(fused_launch.fc1_tile_k),
                "fc2_tile_n": int(fused_launch.fc2_tile_n),
                "fc2_tile_k": int(fused_launch.fc2_tile_k),
                "moe_block_size": int(fused_launch.moe_block_size),
                "max_m_blocks": int(fused_launch.max_m_blocks),
                "blocks_per_sm": int(fused_launch.blocks_per_sm),
                "weight_layout": str(fused_launch.weight_layout),
                "trellis_bits": int(fused_launch.trellis_bits),
                "fc1_trellis_pair_kind": fused_launch.fc1_trellis_pair_kind,
                "fc2_trellis_pair_kind": fused_launch.fc2_trellis_pair_kind,
                "schedule_whole_tiles": bool(fused_launch.schedule_whole_tiles),
                "direct_topk_routes": bool(fused_launch.direct_topk_routes),
                "tc_decode_fused_sum": bool(fused_launch.tc_decode_fused_sum),
            }
        )
    keepalive: list[object] = [
        weights,
        topk_ids,
        topk_weights,
        expert_map,
        x,
        plan,
        scratch,
        binding,
        eager,
        captured,
    ]
    return (
        graph,
        keepalive,
        mode_summary,
        kernel_summary,
        correctness_summary,
        route_replay,
    )


def _interleaved_times_us(
    graphs: dict[str, torch.cuda.CUDAGraph],
    *,
    replays: int,
    flush_l2: Callable[[], None] | None,
    route_replays: dict[str, _RouteReplay | None] | None = None,
    replay_offset: int = 0,
) -> dict[str, list[float]]:
    times: dict[str, list[float]] = {name: [] for name in graphs}
    events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
    names = list(graphs)
    for replay in range(replays):
        order = names if replay % 2 == 0 else list(reversed(names))
        for name in order:
            if route_replays is not None and route_replays[name] is not None:
                route_replays[name].load(replay + replay_offset)
            if flush_l2 is not None:
                flush_l2()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            graphs[name].replay()
            end.record()
            events.append((name, start, end))
    torch.cuda.synchronize()
    for name, start, end in events:
        times[name].append(start.elapsed_time(end) * 1000.0)
    return times


def _summary(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "median_us": statistics.median(ordered),
        "p10_us": ordered[max(0, int(0.10 * (len(ordered) - 1)))],
        "p90_us": ordered[min(len(ordered) - 1, int(0.90 * (len(ordered) - 1)))],
        "min_us": ordered[0],
        "max_us": ordered[-1],
    }


def _paired_ratio_bootstrap(
    samples: list[float],
    baseline: list[float],
    *,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    if len(samples) != len(baseline) or not samples:
        raise ValueError("paired timing samples must have the same nonzero length")
    sample_tensor = torch.tensor(samples, dtype=torch.float64)
    baseline_tensor = torch.tensor(baseline, dtype=torch.float64)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    indices = torch.randint(
        len(samples),
        (replicates, len(samples)),
        dtype=torch.int64,
        generator=generator,
    )
    sample_medians = torch.quantile(sample_tensor[indices], 0.5, dim=1)
    baseline_medians = torch.quantile(baseline_tensor[indices], 0.5, dim=1)
    ratios = sample_medians / baseline_medians
    bounds = torch.quantile(
        ratios, torch.tensor([0.025, 0.975], dtype=torch.float64)
    )
    return {
        "point_estimate": statistics.median(samples)
        / statistics.median(baseline),
        "bootstrap_ci95_low": float(bounds[0]),
        "bootstrap_ci95_high": float(bounds[1]),
        "bootstrap_replicates": replicates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--codebook",
        choices=tuple(_CODEBOOK_SOURCE_FORMATS),
        default="sqg_xor_cheb_t12",
        help="SQG-XOR-Cheb-T12 reconstruction used by every trellis scenario",
    )
    parser.add_argument(
        "--tokens", type=_parse_positive_list, default=_parse_positive_list("1,4,16")
    )
    parser.add_argument(
        "--experts",
        type=int,
        default=None,
        help="materialized experts (default: fixture count, otherwise 32)",
    )
    parser.add_argument(
        "--route-fixture",
        type=Path,
        help=(
            "kquant TP12 fixture with natural routes and exact sparse P24/P33 "
            "mode tables"
        ),
    )
    parser.add_argument(
        "--sparse-p24-experts",
        type=int,
        default=1,
        help="P24 expert prefix used by the sparse production-like scenario",
    )
    parser.add_argument(
        "--scenarios",
        type=_parse_scenarios,
        default=_parse_scenarios(",".join(_SCENARIOS)),
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--replays", type=int, default=200)
    parser.add_argument("--cold-replays", type=int, default=50)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument(
        "--require-known-resources",
        action="store_true",
        help="fail unless CUDA function attributes are available for every case",
    )
    parser.add_argument(
        "--require-no-local-memory",
        action="store_true",
        help="fail unless every fused kernel reports zero local bytes/thread",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    route_fixture = (
        _load_route_fixture(args.route_fixture)
        if args.route_fixture is not None
        else None
    )
    if route_fixture is not None:
        if args.experts is not None and args.experts != route_fixture.num_experts:
            raise SystemExit(
                f"experts={args.experts} disagrees with route fixture "
                f"experts={route_fixture.num_experts}"
            )
        experts = route_fixture.num_experts
        missing_scenarios = sorted({"P33", "sparse"} - set(args.scenarios))
        if missing_scenarios:
            raise SystemExit(
                "route-fixture benchmarking requires scenarios P33,sparse; "
                f"missing {missing_scenarios}"
            )
        max_tokens = max(args.tokens)
        if int(route_fixture.topk_ids.shape[0]) < max_tokens:
            raise SystemExit(
                f"route fixture has {route_fixture.topk_ids.shape[0]} rows, "
                f"fewer than max tokens={max_tokens}"
            )
    else:
        experts = 32 if args.experts is None else args.experts

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    major, minor = torch.cuda.get_device_capability()
    if major != 12 or minor not in (0, 1):
        raise SystemExit(f"SM120/SM121 is required, got SM{major}{minor}")
    if experts < _TOPK or experts > _ROUTE_EXPERTS:
        raise SystemExit(
            f"experts must be in [{_TOPK}, {_ROUTE_EXPERTS}], got {experts}"
        )
    if not 1 <= args.sparse_p24_experts <= experts:
        raise SystemExit(
            "sparse-p24-experts must be in [1, experts], got "
            f"{args.sparse_p24_experts}"
        )
    if (
        args.warmup < 1
        or args.replays < 1
        or args.cold_replays < 0
        or args.bootstrap_replicates < 1
    ):
        raise SystemExit(
            "warmup/replays/bootstrap-replicates must be positive and "
            "cold-replays non-negative"
        )

    device = torch.device("cuda", torch.cuda.current_device())
    props = torch.cuda.get_device_properties(device)
    result: dict[str, object] = {
        "kind": _RESULT_KIND,
        "schema_version": _RESULT_SCHEMA_VERSION,
        "provenance": {
            "command": shlex.join([sys.executable, *sys.argv]),
            "commit": _git_value("rev-parse", "HEAD"),
            "branch": _git_value("branch", "--show-current"),
            "git_dirty": bool(_git_value("status", "--porcelain")),
            "source_sha256": _source_sha256(),
            "worktree": str(Path(__file__).resolve().parents[1]),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "route_fixture": (
                None
                if route_fixture is None
                else {
                    "path": str(route_fixture.path),
                    "sha256": route_fixture.sha256,
                    "manifest": route_fixture.manifest,
                }
            ),
        },
        "contract": {
            "tp": 12,
            "hidden_size": _HIDDEN,
            "local_intermediate_size": _LOCAL_INTERMEDIATE,
            "route_experts": _ROUTE_EXPERTS,
            "topk": _TOPK,
            "materialized_experts": experts,
            "sparse_p24_experts": (
                args.sparse_p24_experts if route_fixture is None else None
            ),
            "route_source": (
                "synthetic" if route_fixture is None else "captured_fixture"
            ),
            "timed_route_window_rotation": route_fixture is not None,
            "route_copy_in_timed_region": False,
            "payload_bytes_per_projection": _PAYLOAD_BYTES_PER_PROJECTION,
            "payload_bytes_per_expert": 3 * _PAYLOAD_BYTES_PER_PROJECTION,
            "codebook": args.codebook,
            "tokens": args.tokens,
            "scenarios": args.scenarios,
            "complete_routed_pipeline": True,
            "outer_hadamard_included": True,
            "cuda_graph_replay": True,
            "cuda_graph_replay_allocation_stable_required": True,
            "paired_timing_bootstrap_replicates": args.bootstrap_replicates,
            "correctness": "finite eager output and bit-exact eager/graph replay",
            "kernel_resource_scope": "fused route/FC1/SiTU/FC2 kernel only",
        },
        "device": {
            "name": props.name,
            "uuid": str(getattr(props, "uuid", "")),
            "capability": [major, minor],
            "torch": torch.__version__,
            "multiprocessor_count": int(props.multi_processor_count),
            "registers_per_multiprocessor": int(props.regs_per_multiprocessor),
            "max_threads_per_multiprocessor": int(
                props.max_threads_per_multi_processor
            ),
            "shared_memory_per_multiprocessor": int(
                props.shared_memory_per_multiprocessor
            ),
            "shared_memory_per_block_optin": int(
                props.shared_memory_per_block_optin
            ),
            "warp_size": int(props.warp_size),
            "mode_before": nvidia_smi_gpu_mode_snapshot(),
        },
        "cases": [],
    }
    cold_flush = make_l2_flush_fn(args.cold_replays > 0)
    all_keepalive: list[object] = []
    cases = result["cases"]
    assert isinstance(cases, list)

    for tokens in args.tokens:
        graphs: dict[str, torch.cuda.CUDAGraph] = {}
        mode_summaries: dict[str, dict[str, float]] = {}
        kernel_summaries: dict[str, dict[str, object]] = {}
        correctness_summaries: dict[str, dict[str, object]] = {}
        route_replays: dict[str, _RouteReplay | None] = {}
        # Keep payload bytes, activations, and routes paired across scenarios;
        # only the format descriptor and its decode path should differ.
        for scenario in args.scenarios:
            (
                graph,
                keepalive,
                mode_summary,
                kernel_summary,
                correctness_summary,
                route_replay,
            ) = _capture_case(
                scenario,
                tokens,
                experts,
                codebook=args.codebook,
                device=device,
                seed=args.seed + tokens * 100,
                warmup=args.warmup,
                sparse_p24_experts=args.sparse_p24_experts,
                route_fixture=route_fixture,
            )
            graphs[scenario] = graph
            route_replays[scenario] = route_replay
            all_keepalive.extend(keepalive)
            mode_summaries[scenario] = mode_summary
            kernel_summaries[scenario] = kernel_summary
            correctness_summaries[scenario] = correctness_summary
            resources_known = bool(
                kernel_summary["resource_introspection_available"]
            )
            local_memory_bytes = int(
                kernel_summary["local_memory_bytes_per_thread"]
            )
            if args.require_known_resources and not resources_known:
                raise RuntimeError(
                    f"{scenario}/tokens={tokens} has unavailable CUDA resource data; "
                    "use a fresh B12X_COMPILE_CACHE_DIR"
                )
            if args.require_no_local_memory and local_memory_bytes != 0:
                raise RuntimeError(
                    f"{scenario}/tokens={tokens} reports "
                    f"{local_memory_bytes} local bytes/thread"
                )

        allocated_before_timing = int(torch.cuda.memory_allocated(device))
        reserved_before_timing = int(torch.cuda.memory_reserved(device))
        hot = _interleaved_times_us(
            graphs,
            replays=args.replays,
            flush_l2=None,
            route_replays=route_replays,
        )
        cold = (
            _interleaved_times_us(
                graphs,
                replays=args.cold_replays,
                flush_l2=cold_flush,
                route_replays=route_replays,
                replay_offset=args.replays,
            )
            if args.cold_replays
            else None
        )
        allocated_after_timing = int(torch.cuda.memory_allocated(device))
        reserved_after_timing = int(torch.cuda.memory_reserved(device))
        if allocated_before_timing != allocated_after_timing:
            raise RuntimeError(
                f"tokens={tokens} CUDA graph replay changed Torch allocation: "
                f"{allocated_before_timing} -> {allocated_after_timing} bytes"
            )
        hot_summary = {name: _summary(values) for name, values in hot.items()}
        cold_summary = (
            {name: _summary(values) for name, values in cold.items()}
            if cold is not None
            else None
        )
        baseline = hot_summary.get("P33")
        if baseline is None:
            raise RuntimeError("P33 must be included as the benchmark baseline")
        baseline_name = "P33"
        hot_ratio_bootstrap = {
            name: _paired_ratio_bootstrap(
                values,
                hot[baseline_name],
                replicates=args.bootstrap_replicates,
                seed=args.seed + tokens * 1000 + index,
            )
            for index, (name, values) in enumerate(hot.items())
        }
        cold_ratio_bootstrap = (
            {
                name: _paired_ratio_bootstrap(
                    values,
                    cold[baseline_name],
                    replicates=args.bootstrap_replicates,
                    seed=args.seed + 500_000 + tokens * 1000 + index,
                )
                for index, (name, values) in enumerate(cold.items())
            }
            if cold is not None
            else None
        )
        case = {
            "tokens": tokens,
            "mode_summary": mode_summaries,
            "kernel_summary": kernel_summaries,
            "correctness": correctness_summaries,
            "hot": hot_summary,
            "raw_hot_us": hot,
            "hot_median_over_baseline": {
                name: summary["median_us"] / baseline["median_us"]
                for name, summary in hot_summary.items()
            },
            "hot_ratio_over_baseline_bootstrap": hot_ratio_bootstrap,
            "cold": cold_summary,
            "raw_cold_us": cold,
            "cold_ratio_over_baseline_bootstrap": cold_ratio_bootstrap,
            "graph_replay_allocation": {
                "allocated_before_bytes": allocated_before_timing,
                "allocated_after_bytes": allocated_after_timing,
                "allocated_stable": (
                    allocated_before_timing == allocated_after_timing
                ),
                "reserved_before_bytes": reserved_before_timing,
                "reserved_after_bytes": reserved_after_timing,
                "reserved_stable": reserved_before_timing == reserved_after_timing,
            },
        }
        cases.append(case)
        print(json.dumps(case, sort_keys=True))

    device_result = result["device"]
    assert isinstance(device_result, dict)
    device_result["mode_after"] = nvidia_smi_gpu_mode_snapshot()
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        _write_new_result(args.output, encoded)
    else:
        print(encoded)


if __name__ == "__main__":
    main()
