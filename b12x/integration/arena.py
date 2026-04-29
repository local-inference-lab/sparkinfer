"""Joint b12x scratch arena APIs.

The execution-lane arena owns one uint8 backing allocation and overlays phase
scratch for MLA attention, paged attention, and MoE. A lane is the unit of true
concurrent scratch ownership; internal fork/join streams within the lane share
this arena.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from b12x.attention.mla.workspace import (
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
    plan_tp_moe_arena_layout,
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
    weight_E: int
    k: int
    n: int
    num_topk: int
    max_tokens: int
    core_token_counts: tuple[int, ...] | None = None
    route_num_experts: int | None = None
    route_logits_dtype: torch.dtype | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "device", _canonical_device(self.device))
        object.__setattr__(self, "weight_E", max(int(self.weight_E), 1))
        object.__setattr__(self, "k", max(int(self.k), 1))
        object.__setattr__(self, "n", max(int(self.n), 1))
        object.__setattr__(self, "num_topk", max(int(self.num_topk), 1))
        object.__setattr__(self, "max_tokens", max(int(self.max_tokens), 1))
        if self.core_token_counts is not None:
            object.__setattr__(
                self,
                "core_token_counts",
                tuple(max(int(token_count), 1) for token_count in self.core_token_counts),
            )
        if self.route_num_experts is not None:
            object.__setattr__(self, "route_num_experts", max(int(self.route_num_experts), 1))

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
        )


@dataclass(frozen=True, kw_only=True)
class B12XJointArenaSpec:
    device: torch.device
    attention_caps: B12XAttentionArenaCaps | None = None
    paged_attention_caps: PagedAttentionArenaCaps | None = None
    moe_caps: B12XMoEArenaCaps | None = None

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
        if self.moe_caps is not None and self.moe_caps.device != device:
            raise ValueError(
                f"MoE caps device {self.moe_caps.device} does not match joint arena device {device}"
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
        moe_layout = spec.moe_caps.layout() if spec.moe_caps is not None else None
        moe_nbytes = moe_layout.total_nbytes if moe_layout is not None else 0
        shared_arena_nbytes = max(attention_nbytes, paged_attention_nbytes, moe_nbytes, 1)
        shared_arena = torch.empty(
            shared_arena_nbytes,
            dtype=torch.uint8,
            device=spec.device,
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
    if lane.arena.spec != spec:
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
