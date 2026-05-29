"""Joint b12x scratch arena APIs.

The execution-lane arena owns one uint8 backing allocation and overlays phase
scratch for MLA attention, paged attention, and MoE. A lane is the unit of true
concurrent scratch ownership; internal fork/join streams within the lane share
this arena.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from b12x.attention.workspace import (
    B12XAttentionArena,
    B12XAttentionArenaCaps,
    B12XAttentionWorkspace,
    B12XAttentionWorkspaceContract,
)
from b12x.attention.paged.workspace import (
    PagedAttentionArena,
    PagedAttentionArenaCaps,
    PagedAttentionWorkspace,
    PagedAttentionWorkspaceContract,
)
from b12x.integration.tp_moe import (
    TPMoEArenaLayout,
    TPMoEWorkspacePool,
    allocate_tp_moe_workspace_pool,
    default_moe_quant_mode,
    materialize_tp_moe_arena_workspaces,
    plan_tp_moe_arena_layout,
)

logger = logging.getLogger(__name__)


def _format_nbytes(nbytes: int) -> str:
    value = float(nbytes)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(value) < 1024.0 or unit == "GiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{nbytes} B"


def _cuda_memory_stats(device: torch.device) -> tuple[int | None, int | None]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return None, None
    return (
        int(torch.cuda.memory_allocated(device)),
        int(torch.cuda.memory_reserved(device)),
    )


def _canonical_device(device: torch.device | str) -> torch.device:
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device


def _device_key(device: torch.device | str) -> tuple[torch.device, int]:
    device = _canonical_device(device)
    if device.type == "cuda":
        return device, int(device.index)
    return device, -1


@dataclass(frozen=True, kw_only=True)
class B12XMoEArenaCaps:
    device: torch.device
    dtype: torch.dtype
    quant_mode: str | None = None
    source_format: str = "modelopt_nvfp4"
    weight_E: int
    k: int
    n: int
    num_topk: int
    max_tokens: int
    core_token_counts: tuple[int, ...] | None = None
    route_num_experts: int | None = None
    route_logits_dtype: torch.dtype | None = None
    activation: str = "silu"
    apply_router_weight_on_input: bool = False
    swiglu_limit: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "device", _canonical_device(self.device))
        quant_mode = (
            default_moe_quant_mode()
            if self.quant_mode is None
            else str(self.quant_mode).lower()
        )
        if quant_mode not in {"nvfp4", "w4a16"}:
            raise ValueError(f"unsupported quant_mode {self.quant_mode!r}")
        object.__setattr__(self, "quant_mode", quant_mode)
        object.__setattr__(self, "source_format", str(self.source_format).lower())
        object.__setattr__(self, "weight_E", max(int(self.weight_E), 1))
        object.__setattr__(self, "k", max(int(self.k), 1))
        object.__setattr__(self, "n", max(int(self.n), 1))
        object.__setattr__(self, "num_topk", max(int(self.num_topk), 1))
        object.__setattr__(self, "max_tokens", max(int(self.max_tokens), 1))
        activation = str(self.activation).lower()
        if activation not in {"silu", "relu2"}:
            raise ValueError(f"unsupported activation {self.activation!r}")
        object.__setattr__(self, "activation", activation)
        object.__setattr__(
            self,
            "apply_router_weight_on_input",
            bool(self.apply_router_weight_on_input),
        )
        if self.swiglu_limit is not None:
            object.__setattr__(self, "swiglu_limit", float(self.swiglu_limit))
        if self.core_token_counts is not None:
            object.__setattr__(
                self,
                "core_token_counts",
                tuple(
                    max(int(token_count), 1)
                    for token_count in self.core_token_counts
                ),
            )
        if self.route_num_experts is not None:
            object.__setattr__(
                self, "route_num_experts", max(int(self.route_num_experts), 1)
            )

    def layout(self) -> TPMoEArenaLayout:
        return plan_tp_moe_arena_layout(
            max_tokens=self.max_tokens,
            weight_E=self.weight_E,
            k=self.k,
            n=self.n,
            num_topk=self.num_topk,
            device=self.device,
            dtype=self.dtype,
            core_token_counts=self.core_token_counts,
            route_num_experts=self.route_num_experts,
            route_logits_dtype=self.route_logits_dtype,
            quant_mode=self.quant_mode,
            activation=self.activation,
        )


def _moe_caps_tuple(
    caps: B12XMoEArenaCaps | tuple[B12XMoEArenaCaps, ...] | None,
) -> tuple[B12XMoEArenaCaps, ...]:
    if caps is None:
        return ()
    if isinstance(caps, B12XMoEArenaCaps):
        return (caps,)
    return tuple(caps)


def _combined_moe_layout(
    caps: tuple[B12XMoEArenaCaps, ...],
) -> TPMoEArenaLayout | None:
    if not caps:
        return None
    layouts = tuple(cap.layout() for cap in caps)
    route_nbytes = max(layout.route_workspace_nbytes for layout in layouts)
    core_nbytes = max(layout.core_workspace_nbytes for layout in layouts)
    core_token_counts = tuple(
        sorted(
            {
                token_count
                for layout in layouts
                for token_count in layout.core_token_counts
            }
        )
    )
    return TPMoEArenaLayout(
        route_workspace_nbytes=route_nbytes,
        core_workspace_nbytes=core_nbytes,
        total_nbytes=max(route_nbytes + core_nbytes, 1),
        core_token_counts=core_token_counts,
    )


@dataclass(frozen=True, kw_only=True)
class B12XJointArenaSpec:
    device: torch.device
    attention_caps: B12XAttentionArenaCaps | None = None
    paged_attention_caps: PagedAttentionArenaCaps | None = None
    moe_caps: B12XMoEArenaCaps | tuple[B12XMoEArenaCaps, ...] | None = None

    def __post_init__(self) -> None:
        device = _canonical_device(self.device)
        object.__setattr__(self, "device", device)
        if self.attention_caps is not None and self.attention_caps.device != device:
            raise ValueError(
                f"attention caps device {self.attention_caps.device} does not match joint arena device {device}"
            )
        if self.paged_attention_caps is not None and self.paged_attention_caps.device != device:
            raise ValueError(
                "paged attention caps device "
                f"{self.paged_attention_caps.device} does not match joint arena device {device}"
            )
        moe_caps = _moe_caps_tuple(self.moe_caps)
        for caps in moe_caps:
            if caps.device != device:
                raise ValueError(
                    f"MoE caps device {caps.device} does not match joint arena device {device}"
                )
        object.__setattr__(self, "moe_caps", moe_caps or None)


def _field_matches(existing: object, requested: object, field: str) -> bool:
    return getattr(existing, field) == getattr(requested, field)


def _fields_match(existing: object, requested: object, fields: tuple[str, ...]) -> bool:
    return all(_field_matches(existing, requested, field) for field in fields)


def _fields_cover(existing: object, requested: object, fields: tuple[str, ...]) -> bool:
    return all(getattr(existing, field) >= getattr(requested, field) for field in fields)


def _attention_caps_cover(
    existing: B12XAttentionArenaCaps | None,
    requested: B12XAttentionArenaCaps | None,
) -> bool:
    if requested is None:
        return True
    if existing is None:
        return False
    if not _fields_match(
        existing,
        requested,
        (
            "device",
            "dtype",
            "kv_dtype",
            "num_q_heads",
            "head_dim",
            "page_size",
            "padded_heads",
        ),
    ):
        return False
    if requested.reserve_extend_indexer_logits and not existing.reserve_extend_indexer_logits:
        return False
    if requested.reserve_paged_indexer_logits and not existing.reserve_paged_indexer_logits:
        return False
    if getattr(requested, "reserve_mhc", False) and not getattr(existing, "reserve_mhc", False):
        return False
    if getattr(requested, "reserve_mhc", False) and (
        getattr(existing, "mhc_hidden_size", 0) != getattr(requested, "mhc_hidden_size", 0)
        or getattr(existing, "mhc_split_k", 0) != getattr(requested, "mhc_split_k", 0)
    ):
        return False
    if getattr(requested, "reserve_wo_projection", False) and not getattr(
        existing,
        "reserve_wo_projection",
        False,
    ):
        return False
    if getattr(requested, "reserve_wo_projection", False) and (
        getattr(existing, "wo_groups", 0) != getattr(requested, "wo_groups", 0)
        or getattr(existing, "wo_group_width", 0)
        != getattr(requested, "wo_group_width", 0)
        or getattr(existing, "wo_rank", 0) != getattr(requested, "wo_rank", 0)
        or getattr(existing, "wo_hidden_size", 0)
        != getattr(requested, "wo_hidden_size", 0)
    ):
        return False
    existing_mla_q_chunks = int(
        getattr(existing, "mla_max_q_chunks", 0)
        or int(getattr(existing, "mla_max_total_q", 1))
        * int(getattr(existing, "max_chunks_per_row", 1))
    )
    requested_mla_q_chunks = int(
        getattr(requested, "mla_max_q_chunks", 0)
        or int(getattr(requested, "mla_max_total_q", 1))
        * int(getattr(requested, "max_chunks_per_row", 1))
    )
    if existing_mla_q_chunks < requested_mla_q_chunks:
        return False
    return _fields_cover(
        existing,
        requested,
        (
            "indexer_num_q_heads",
            "max_v_head_dim",
            "topk",
            "indexer_topk",
            "max_page_table_width",
            "extend_max_total_q",
            "extend_max_batch",
            "extend_max_kv_rows",
            "indexer_max_k_rows",
            "paged_max_q_rows",
            "paged_max_batch",
            "mla_max_total_q",
            "max_chunks_per_row",
            "extend_indexer_tile_logits_k_rows",
            "paged_indexer_logits_k_rows",
            "paged_indexer_tile_logits_k_rows",
            "mhc_max_tokens",
            "wo_max_tokens",
        ),
    )


def _paged_attention_caps_cover(
    existing: PagedAttentionArenaCaps | None,
    requested: PagedAttentionArenaCaps | None,
) -> bool:
    if requested is None:
        return True
    if existing is None:
        return False
    if not _fields_match(
        existing,
        requested,
        ("device", "dtype", "kv_dtype", "page_size"),
    ):
        return False
    return _fields_cover(
        existing,
        requested,
        (
            "num_q_heads",
            "num_kv_heads",
            "head_dim_qk",
            "max_head_dim_vo",
            "max_total_q",
            "max_batch",
            "max_page_table_width",
            "max_work_items",
            "max_partial_rows",
        ),
    )


def _moe_route_num_experts(caps: B12XMoEArenaCaps) -> int:
    return int(
        caps.route_num_experts if caps.route_num_experts is not None else caps.weight_E
    )


def _moe_route_logits_dtype(caps: B12XMoEArenaCaps) -> torch.dtype:
    return caps.route_logits_dtype or caps.dtype


def _single_moe_caps_cover(
    existing: B12XMoEArenaCaps,
    requested: B12XMoEArenaCaps,
) -> bool:
    if not _fields_match(
        existing,
        requested,
        (
            "device",
            "dtype",
            "quant_mode",
            "source_format",
            "activation",
            "apply_router_weight_on_input",
            "swiglu_limit",
            "weight_E",
            "k",
            "n",
            "num_topk",
        ),
    ):
        return False
    if _moe_route_num_experts(existing) != _moe_route_num_experts(requested):
        return False
    if _moe_route_logits_dtype(existing) != _moe_route_logits_dtype(requested):
        return False
    existing_layout = existing.layout()
    requested_layout = requested.layout()
    if existing.quant_mode == "w4a16" and not set(
        requested_layout.core_token_counts
    ).issubset(set(existing_layout.core_token_counts)):
        return False
    return (
        existing_layout.route_workspace_nbytes >= requested_layout.route_workspace_nbytes
        and existing_layout.core_workspace_nbytes >= requested_layout.core_workspace_nbytes
    )


def _moe_caps_cover(
    existing: B12XMoEArenaCaps | tuple[B12XMoEArenaCaps, ...] | None,
    requested: B12XMoEArenaCaps | tuple[B12XMoEArenaCaps, ...] | None,
) -> bool:
    requested_caps = _moe_caps_tuple(requested)
    if not requested_caps:
        return True
    existing_caps = _moe_caps_tuple(existing)
    if not existing_caps:
        return False
    return all(
        any(
            _single_moe_caps_cover(existing_cap, requested_cap)
            for existing_cap in existing_caps
        )
        for requested_cap in requested_caps
    )


def _joint_arena_spec_covers(
    existing: B12XJointArenaSpec,
    requested: B12XJointArenaSpec,
) -> bool:
    if existing.device != requested.device:
        return False
    return (
        _attention_caps_cover(existing.attention_caps, requested.attention_caps)
        and _paged_attention_caps_cover(
            existing.paged_attention_caps,
            requested.paged_attention_caps,
        )
        and _moe_caps_cover(existing.moe_caps, requested.moe_caps)
    )


@dataclass(kw_only=True)
class B12XExecutionLaneArena:
    spec: B12XJointArenaSpec
    shared_arena: torch.Tensor
    shared_arena_nbytes: int
    attention_nbytes: int = 0
    paged_attention_nbytes: int = 0
    moe_nbytes: int = 0
    moe_layout: TPMoEArenaLayout | None = None
    attention_arena: B12XAttentionArena | None = None
    paged_attention_arena: PagedAttentionArena | None = None
    moe_workspace_pool: TPMoEWorkspacePool | None = None

    @classmethod
    def allocate(cls, spec: B12XJointArenaSpec) -> "B12XExecutionLaneArena":
        moe_caps = _moe_caps_tuple(spec.moe_caps)
        attention_nbytes = (
            B12XAttentionArena.required_nbytes(spec.attention_caps)
            if spec.attention_caps is not None
            else 0
        )
        paged_attention_nbytes = (
            PagedAttentionArena.required_nbytes(spec.paged_attention_caps)
            if spec.paged_attention_caps is not None
            else 0
        )
        moe_layout = _combined_moe_layout(moe_caps)
        moe_nbytes = moe_layout.total_nbytes if moe_layout is not None else 0
        shared_arena_nbytes = max(
            attention_nbytes,
            paged_attention_nbytes,
            moe_nbytes,
            1,
        )
        allocated_before, reserved_before = _cuda_memory_stats(spec.device)
        logger.warning(
            "B12X joint arena request: device=%s shared=%s (%d bytes), "
            "attention_required=%s, paged_attention_required=%s, moe_required=%s, "
            "moe_route=%s, moe_core=%s, cuda_allocated_before=%s, cuda_reserved_before=%s",
            spec.device,
            _format_nbytes(shared_arena_nbytes),
            shared_arena_nbytes,
            _format_nbytes(attention_nbytes),
            _format_nbytes(paged_attention_nbytes),
            _format_nbytes(moe_nbytes),
            _format_nbytes(moe_layout.route_workspace_nbytes)
            if moe_layout is not None
            else "0.00 B",
            _format_nbytes(moe_layout.core_workspace_nbytes)
            if moe_layout is not None
            else "0.00 B",
            _format_nbytes(allocated_before) if allocated_before is not None else "n/a",
            _format_nbytes(reserved_before) if reserved_before is not None else "n/a",
        )
        shared_arena = torch.empty(
            shared_arena_nbytes,
            dtype=torch.uint8,
            device=spec.device,
        )
        allocated_after, reserved_after = _cuda_memory_stats(spec.device)
        allocated_delta = (
            allocated_after - allocated_before
            if allocated_before is not None and allocated_after is not None
            else None
        )
        reserved_delta = (
            reserved_after - reserved_before
            if reserved_before is not None and reserved_after is not None
            else None
        )
        logger.warning(
            "B12X joint arena allocation: device=%s shared=%s (%d bytes), "
            "attention_required=%s, paged_attention_required=%s, moe_required=%s, "
            "moe_route=%s, moe_core=%s, cuda_allocated_delta=%s, "
            "cuda_reserved_delta=%s, cuda_allocated=%s, cuda_reserved=%s",
            spec.device,
            _format_nbytes(shared_arena_nbytes),
            shared_arena_nbytes,
            _format_nbytes(attention_nbytes),
            _format_nbytes(paged_attention_nbytes),
            _format_nbytes(moe_nbytes),
            _format_nbytes(moe_layout.route_workspace_nbytes)
            if moe_layout is not None
            else "0.00 B",
            _format_nbytes(moe_layout.core_workspace_nbytes)
            if moe_layout is not None
            else "0.00 B",
            _format_nbytes(allocated_delta) if allocated_delta is not None else "n/a",
            _format_nbytes(reserved_delta) if reserved_delta is not None else "n/a",
            _format_nbytes(allocated_after) if allocated_after is not None else "n/a",
            _format_nbytes(reserved_after) if reserved_after is not None else "n/a",
        )

        attention_arena = (
            B12XAttentionArena.from_shared_arena(spec.attention_caps, shared_arena)
            if spec.attention_caps is not None
            else None
        )
        paged_attention_arena = (
            PagedAttentionArena.from_shared_arena(spec.paged_attention_caps, shared_arena)
            if spec.paged_attention_caps is not None
            else None
        )
        moe_workspace_pool = None
        if moe_layout is not None:
            moe_workspace_pool = allocate_tp_moe_workspace_pool(
                shared_arena=shared_arena,
                route_workspace_nbytes=moe_layout.route_workspace_nbytes,
                core_workspace_nbytes=moe_layout.core_workspace_nbytes,
                frozen=True,
            )
            for caps in moe_caps:
                materialize_tp_moe_arena_workspaces(
                    moe_workspace_pool,
                    max_tokens=caps.max_tokens,
                    weight_E=caps.weight_E,
                    k=caps.k,
                    n=caps.n,
                    num_topk=caps.num_topk,
                    device=caps.device,
                    dtype=caps.dtype,
                    core_token_counts=caps.core_token_counts,
                    quant_mode=caps.quant_mode,
                    source_format=caps.source_format,
                    w13_layout=caps.w13_layout,
                    activation=caps.activation,
                    apply_router_weight_on_input=caps.apply_router_weight_on_input,
                    swiglu_limit=caps.swiglu_limit,
                )

        lane = cls(
            spec=spec,
            shared_arena=shared_arena,
            shared_arena_nbytes=shared_arena_nbytes,
            attention_nbytes=attention_nbytes,
            paged_attention_nbytes=paged_attention_nbytes,
            moe_nbytes=moe_nbytes,
            moe_layout=moe_layout,
            attention_arena=attention_arena,
            paged_attention_arena=paged_attention_arena,
            moe_workspace_pool=moe_workspace_pool,
        )
        return lane

    def make_attention_workspace(
        self,
        contract: B12XAttentionWorkspaceContract,
        *,
        use_cuda_graph: bool = False,
    ) -> B12XAttentionWorkspace:
        if self.attention_arena is None:
            raise RuntimeError("execution lane arena was allocated without attention caps")
        return self.attention_arena.make_workspace(contract, use_cuda_graph=use_cuda_graph)

    def make_paged_attention_workspace(
        self,
        contract: PagedAttentionWorkspaceContract,
        *,
        use_cuda_graph: bool = False,
    ) -> PagedAttentionWorkspace:
        if self.paged_attention_arena is None:
            raise RuntimeError("execution lane arena was allocated without paged attention caps")
        return self.paged_attention_arena.make_workspace(
            contract,
            use_cuda_graph=use_cuda_graph,
        )

    def get_moe_workspace_pool(self) -> TPMoEWorkspacePool:
        if self.moe_workspace_pool is None:
            raise RuntimeError("execution lane arena was allocated without MoE caps")
        return self.moe_workspace_pool


@dataclass
class B12XExecutionLane:
    """Process-local scratch ownership for one execution lane."""

    device: torch.device
    moe_workspace_pool: TPMoEWorkspacePool
    arena: B12XExecutionLaneArena | None = None


_EXECUTION_LANES: dict[int, B12XExecutionLane] = {}


def _install_b12x_execution_lane_arena(
    device: torch.device | str,
    arena: B12XExecutionLaneArena,
) -> B12XExecutionLane:
    canonical_device, device_idx = _device_key(device)
    lane = B12XExecutionLane(
        device=canonical_device,
        moe_workspace_pool=arena.get_moe_workspace_pool()
        if arena.moe_workspace_pool is not None
        else allocate_tp_moe_workspace_pool(),
        arena=arena,
    )
    _EXECUTION_LANES[device_idx] = lane
    return lane


def set_b12x_execution_lane_arena(
    device: torch.device | str,
    arena: B12XExecutionLaneArena,
) -> B12XExecutionLane:
    """Install a caller-created joint arena as the process-local execution lane."""
    return _install_b12x_execution_lane_arena(device, arena)


def ensure_b12x_execution_lane_arena(spec: B12XJointArenaSpec) -> B12XExecutionLane:
    """Return the device lane, allocating the requested joint arena if needed."""
    canonical_device, device_idx = _device_key(spec.device)
    lane = _EXECUTION_LANES.get(device_idx)
    if lane is None:
        arena = B12XExecutionLaneArena.allocate(spec)
        return _install_b12x_execution_lane_arena(canonical_device, arena)
    if lane.arena is None:
        if lane.moe_workspace_pool.workspaces or lane.moe_workspace_pool.route_workspaces:
            raise RuntimeError(
                "cannot replace an active standalone b12x MoE workspace pool with a joint arena"
            )
        arena = B12XExecutionLaneArena.allocate(spec)
        return _install_b12x_execution_lane_arena(canonical_device, arena)
    if lane.arena.spec != spec and not _joint_arena_spec_covers(lane.arena.spec, spec):
        raise RuntimeError(
            "existing b12x execution lane arena has incompatible sizing caps for this device"
        )
    return lane


def get_b12x_execution_lane(
    device: torch.device | str,
    *,
    create_standalone_moe_pool: bool = True,
) -> B12XExecutionLane | None:
    """Return the process-local b12x execution lane for *device*."""
    canonical_device, device_idx = _device_key(device)
    lane = _EXECUTION_LANES.get(device_idx)
    if lane is None and create_standalone_moe_pool:
        lane = B12XExecutionLane(
            device=canonical_device,
            moe_workspace_pool=allocate_tp_moe_workspace_pool(),
        )
        _EXECUTION_LANES[device_idx] = lane
    return lane


def get_b12x_moe_workspace_pool(device: torch.device | str) -> TPMoEWorkspacePool:
    """Return the MoE workspace pool owned by the device execution lane."""
    lane = get_b12x_execution_lane(device)
    assert lane is not None
    return lane.moe_workspace_pool
