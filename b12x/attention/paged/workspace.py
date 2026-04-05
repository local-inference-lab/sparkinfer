"""Dynamic workspace state for the primary paged-attention backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch

from b12x.profiling import record_function

from .planner import (
    PagedPlan,
    PagedPlanBudget,
    build_decode_chunk_pages_lut,
    create_paged_plan,
    infer_paged_mode,
)


def _paged_lse_storage_shape(total_q: int, num_q_heads: int) -> tuple[int, int]:
    return (num_q_heads, total_q)


def _copy_int_metadata(values: tuple[int, ...], *, device: torch.device) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int32, device=device)


def _canonical_device(device: torch.device | str) -> torch.device:
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device


def _shape_only_cuda_tensor(
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Return a tiny CUDA tensor view with the requested logical shape.

    The paged planner only inspects shape, dtype, and device for the
    contract-side KV tensors; it never reads their data. Represent the
    contract with a single-element zero-stride view so graph workspaces do
    not allocate full shadow copies of the KV cache.
    """

    if any(dim <= 0 for dim in shape):
        raise ValueError(f"shape must be positive in every dimension, got {shape}")
    base = torch.empty(1, dtype=dtype, device=device)
    return base.as_strided(shape, (0,) * len(shape))


@dataclass(kw_only=True)
class PagedAttentionWorkspace:
    mode: Literal["decode", "extend", "verify"]
    device: torch.device
    dtype: torch.dtype
    kv_dtype: torch.dtype
    num_q_heads: int
    num_kv_heads: int
    head_dim_qk: int
    head_dim_vo: int
    attn_mode: Literal["default", "turbo"] | None = None
    page_size: int = 64
    use_cuda_graph: bool = False
    fixed_capacity: bool = False
    request_indices: torch.Tensor | None = None
    qo_tile_indices: torch.Tensor | None = None
    kv_tile_indices: torch.Tensor | None = None
    merge_indptr: torch.Tensor | None = None
    o_indptr: torch.Tensor | None = None
    kv_chunk_size_ptr: torch.Tensor | None = None
    total_num_rows_ptr: torch.Tensor | None = None
    block_valid_mask: torch.Tensor | None = None
    page_table: torch.Tensor | None = None
    cache_seqlens: torch.Tensor | None = None
    cu_seqlens_q: torch.Tensor | None = None
    lse: torch.Tensor | None = None
    tmp_output: torch.Tensor | None = None
    tmp_lse: torch.Tensor | None = None
    _plan_q: torch.Tensor | None = None
    _plan_output: torch.Tensor | None = None
    _plan_k_cache: torch.Tensor | None = None
    _plan_v_cache: torch.Tensor | None = None
    _plan: PagedPlan | None = None
    _planner_budget: PagedPlanBudget | None = None

    # Pre-compiled kernel state (set by compile_paged_kernels in api.py).
    _compiled_forward: object | None = None
    _compiled_merge: object | None = None
    _compiled_forward_kernel: object | None = None
    _compiled_key: tuple | None = None
    _decode_graph_chunk_pages_lut: torch.Tensor | None = None
    _decode_graph_max_chunks_per_req: int | None = None
    _use_regular_decode_graph_replay: bool = False
    _decode_graph_metadata_captured_in_graph: bool = False
    _live_plane_tma_desc_cache: dict[tuple[int, int, tuple[int, ...], tuple[int, ...], int, int], tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor, torch.Tensor]] = field(
        default_factory=dict
    )

    @staticmethod
    def eager_extend_work_items_capacity(
        *,
        max_total_q: int,
        num_q_heads: int,
        num_kv_heads: int,
    ) -> int:
        if max_total_q <= 0:
            raise ValueError("max_total_q must be positive")
        if num_q_heads <= 0 or num_kv_heads <= 0:
            raise ValueError("head counts must be positive")
        gqa_group_size = num_q_heads // num_kv_heads
        return max((int(max_total_q) * int(gqa_group_size) + 15) // 16, 1)

    @classmethod
    def for_contract(
        cls,
        *,
        mode: Literal["decode", "extend", "verify"],
        device: torch.device | str,
        dtype: torch.dtype,
        kv_dtype: torch.dtype,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim_qk: int,
        head_dim_vo: int,
        page_size: int,
        max_total_q: int,
        num_cache_pages: int,
        use_cuda_graph: bool = False,
        attn_mode: Literal["default", "turbo"] | None = None,
    ) -> PagedAttentionWorkspace:
        device = _canonical_device(device)
        if max_total_q <= 0:
            raise ValueError("max_total_q must be positive")
        if num_cache_pages <= 0:
            raise ValueError("num_cache_pages must be positive")
        plan_q = _shape_only_cuda_tensor(
            (max_total_q, num_q_heads, head_dim_qk),
            dtype=dtype,
            device=device,
        )
        plan_output = _shape_only_cuda_tensor(
            (max_total_q, num_q_heads, head_dim_vo),
            dtype=dtype,
            device=device,
        )
        plan_k_cache = _shape_only_cuda_tensor(
            (num_cache_pages, page_size, num_kv_heads, head_dim_qk),
            dtype=kv_dtype,
            device=device,
        )
        plan_v_cache = _shape_only_cuda_tensor(
            (num_cache_pages, page_size, num_kv_heads, head_dim_vo),
            dtype=kv_dtype,
            device=device,
        )
        return cls(
            mode=mode,
            device=device,
            dtype=dtype,
            kv_dtype=kv_dtype,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim_qk,
            head_dim_vo=head_dim_vo,
            attn_mode=attn_mode,
            page_size=page_size,
            use_cuda_graph=use_cuda_graph,
            _plan_q=plan_q,
            _plan_output=plan_output,
            _plan_k_cache=plan_k_cache,
            _plan_v_cache=plan_v_cache,
        )

    @classmethod
    def for_fixed_capacity(
        cls,
        *,
        mode: Literal["decode", "extend", "verify"],
        device: torch.device | str,
        dtype: torch.dtype,
        kv_dtype: torch.dtype,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim_qk: int,
        head_dim_vo: int,
        page_size: int,
        max_total_q: int,
        max_batch: int,
        max_page_table_width: int,
        max_work_items: int,
        max_partial_rows: int,
        num_cache_pages: int,
        use_cuda_graph: bool = False,
        attn_mode: Literal["default", "turbo"] | None = None,
    ) -> PagedAttentionWorkspace:
        workspace = cls.for_contract(
            mode=mode,
            device=device,
            dtype=dtype,
            kv_dtype=kv_dtype,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim_qk,
            head_dim_vo=head_dim_vo,
            page_size=page_size,
            max_total_q=max_total_q,
            num_cache_pages=num_cache_pages,
            use_cuda_graph=use_cuda_graph,
            attn_mode=attn_mode,
        )
        workspace.fixed_capacity = True
        workspace._planner_budget = PagedPlanBudget(
            max_total_q=int(max_total_q),
            max_batch=int(max_batch),
            max_page_table_width=int(max_page_table_width),
            max_work_items=int(max_work_items),
            max_partial_rows=int(max_partial_rows),
        )
        workspace._allocate_runtime_buffers(
            work_items_capacity=int(max_work_items),
            block_valid_capacity=int(max_work_items),
            total_q_capacity=int(max_total_q),
            batch_capacity=int(max_batch),
            page_table_width_capacity=int(max_page_table_width),
            partial_rows_capacity=int(max_partial_rows),
        )
        return workspace

    @classmethod
    def for_eager_extend_capacity(
        cls,
        *,
        mode: Literal["extend", "verify"] = "extend",
        device: torch.device | str,
        dtype: torch.dtype,
        kv_dtype: torch.dtype,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim_qk: int,
        head_dim_vo: int,
        page_size: int,
        max_total_q: int,
        max_batch: int,
        max_page_table_width: int,
        num_cache_pages: int,
        use_cuda_graph: bool = False,
        attn_mode: Literal["default", "turbo"] | None = None,
    ) -> PagedAttentionWorkspace:
        return cls.for_fixed_capacity(
            mode=mode,
            device=device,
            dtype=dtype,
            kv_dtype=kv_dtype,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim_qk,
            head_dim_vo=head_dim_vo,
            page_size=page_size,
            max_total_q=max_total_q,
            max_batch=max_batch,
            max_page_table_width=max_page_table_width,
            max_work_items=cls.eager_extend_work_items_capacity(
                max_total_q=max_total_q,
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
            ),
            max_partial_rows=0,
            num_cache_pages=num_cache_pages,
            use_cuda_graph=use_cuda_graph,
            attn_mode=attn_mode,
        )

    @classmethod
    def for_tensors(
        cls,
        *,
        mode: Literal["decode", "extend", "verify"],
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        use_cuda_graph: bool = False,
        attn_mode: Literal["default", "turbo"] | None = None,
    ) -> PagedAttentionWorkspace:
        if q.ndim != 3:
            raise ValueError(f"q must have shape [total_q, q_heads, head_dim], got {tuple(q.shape)}")
        if k_cache.ndim != 4 or v_cache.ndim != 4:
            raise ValueError("k_cache and v_cache must be rank-4 paged tensors")
        return cls.for_contract(
            mode=mode,
            device=q.device,
            dtype=q.dtype,
            kv_dtype=k_cache.dtype,
            num_q_heads=int(q.shape[1]),
            num_kv_heads=int(k_cache.shape[2]),
            head_dim_qk=int(q.shape[2]),
            head_dim_vo=int(v_cache.shape[3]),
            page_size=int(k_cache.shape[1]),
            max_total_q=int(q.shape[0]),
            num_cache_pages=int(k_cache.shape[0]),
            use_cuda_graph=use_cuda_graph,
            attn_mode=attn_mode,
        )

    @property
    def prepared(self) -> bool:
        return self._plan is not None

    @property
    def plan(self) -> PagedPlan:
        if self._plan is None:
            raise RuntimeError("paged workspace has not been prepared")
        return self._plan

    @property
    def active_total_q(self) -> int:
        return self.plan.total_q

    @property
    def active_batch(self) -> int:
        return self.plan.page_table_shape[0]

    @property
    def total_q_capacity(self) -> int:
        if self._plan_q is None:
            raise RuntimeError("paged workspace planning contract is not initialized")
        return int(self._plan_q.shape[0])

    @property
    def planner_budget(self) -> PagedPlanBudget | None:
        return self._planner_budget

    @staticmethod
    def _plan_has_regular_decode_graph_grid(plan: PagedPlan) -> bool:
        batch = int(plan.page_table_shape[0])
        if batch <= 0 or plan.mode != "decode" or not plan.enable_cuda_graph or not plan.split_kv:
            return False
        if int(plan.total_q) != batch:
            return False
        work_items = len(plan.request_indices)
        if work_items <= 0 or work_items % batch != 0:
            return False
        max_chunks_per_req = work_items // batch
        for req_idx in range(batch):
            base = req_idx * max_chunks_per_req
            for chunk_idx in range(max_chunks_per_req):
                work_idx = base + chunk_idx
                if plan.request_indices[work_idx] != req_idx:
                    return False
                if plan.qo_tile_indices[work_idx] != 0:
                    return False
                if plan.kv_tile_indices[work_idx] != chunk_idx:
                    return False
        return True

    def prepare(
        self,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        *,
        fixed_split_size: int | None = None,
        disable_split_kv: bool = False,
    ) -> PagedAttentionWorkspace:
        with record_function(f"paged_workspace.prepare.{self.mode}"):
            if self.use_cuda_graph and torch.cuda.is_current_stream_capturing():
                if self._plan is None:
                    raise RuntimeError(
                        "graph-mode paged workspace must be prepared before CUDA graph capture"
                    )
                with record_function("paged_workspace.copy_runtime_metadata.capture"):
                    self._copy_runtime_metadata(page_table, cache_seqlens, cu_seqlens_q)
                return self

            if (
                self.use_cuda_graph
                and self.mode == "decode"
                and self._decode_graph_chunk_pages_lut is not None
                and self._plan is not None
            ):
                with record_function("paged_workspace.copy_runtime_metadata"):
                    self._copy_runtime_metadata(page_table, cache_seqlens, cu_seqlens_q)
                if not self._decode_graph_metadata_captured_in_graph:
                    with record_function("paged_workspace.update_decode_graph_replay_metadata"):
                        self.update_decode_graph_replay_metadata_from_runtime_cache_seqlens()
                return self

            with record_function("paged_workspace.infer_mode"):
                inferred_mode = infer_paged_mode(cu_seqlens_q)
            if inferred_mode != self.mode and not (
                self.mode == "extend" and inferred_mode == "decode"
            ) and not (
                self.mode == "verify" and inferred_mode == "extend"
            ):
                raise ValueError(f"workspace mode {self.mode} does not match prepared mode {inferred_mode}")
            active_total_q = int(cu_seqlens_q[-1].item())
            with record_function("paged_workspace.ensure_plan_contract"):
                self._ensure_plan_contract(active_total_q)
            assert self._plan_q is not None
            assert self._plan_k_cache is not None
            assert self._plan_v_cache is not None
            if active_total_q <= 0 or active_total_q > int(self._plan_q.shape[0]):
                raise ValueError(
                    f"cu_seqlens_q implies total_q={active_total_q}, but workspace capacity is {int(self._plan_q.shape[0])}"
                )

            with record_function("paged_workspace.create_paged_plan"):
                plan = create_paged_plan(
                    self._plan_q[:active_total_q],
                    self._plan_k_cache,
                    self._plan_v_cache,
                    page_table,
                    cache_seqlens,
                    cu_seqlens_q,
                    mode=self.mode,
                    fixed_split_size=-1 if fixed_split_size is None else int(fixed_split_size),
                    disable_split_kv=disable_split_kv,
                    enable_cuda_graph=self.use_cuda_graph,
                    graph_chunk_policy=self.use_cuda_graph,
                    plan_budget=(
                        self.planner_budget
                        if self.mode in ("extend", "verify") and not self.use_cuda_graph
                        else None
                    ),
                )
            with record_function("paged_workspace.ensure_capacity"):
                self._ensure_capacity(plan)
            with record_function("paged_workspace.copy_runtime_metadata"):
                self._copy_runtime_metadata(page_table, cache_seqlens, cu_seqlens_q)
            with record_function("paged_workspace.copy_plan_metadata"):
                self._copy_plan_metadata(plan)
            self._plan = plan
            return self

    def update_decode_graph_replay_metadata_from_runtime_cache_seqlens(
        self,
    ) -> PagedAttentionWorkspace:
        if not self.use_cuda_graph:
            raise RuntimeError(
                "update_decode_graph_replay_metadata_from_runtime_cache_seqlens "
                "is only valid for graph-mode workspaces"
            )
        if self.mode != "decode":
            raise RuntimeError(
                "update_decode_graph_replay_metadata_from_runtime_cache_seqlens "
                "is only valid for decode workspaces"
            )
        if self._decode_graph_chunk_pages_lut is None:
            raise RuntimeError("decode graph replay policy has not been prepared")
        if self.cache_seqlens is None:
            raise RuntimeError("decode graph workspace is missing cache_seqlens")
        if self.request_indices is None:
            raise RuntimeError("decode graph workspace is missing request indices")
        if self.merge_indptr is None or self.o_indptr is None:
            raise RuntimeError("decode graph workspace is missing indptr buffers")
        if self.kv_chunk_size_ptr is None:
            raise RuntimeError("decode graph workspace is missing kv_chunk_size_ptr")
        if self.block_valid_mask is None:
            raise RuntimeError("decode graph workspace is missing block_valid_mask")

        self._validate_decode_graph_replay_capacity(batch=int(self.cache_seqlens.shape[0]))

        if self._use_regular_decode_graph_replay:
            from .graph_replay import update_regular_decode_graph_chunk_metadata

            max_cache_pages = torch.div(
                self.cache_seqlens.amax() + (self.page_size - 1),
                self.page_size,
                rounding_mode="floor",
            ).clamp_(min=1, max=self._decode_graph_chunk_pages_lut.shape[0] - 1)
            decode_chunk_pages = torch.index_select(
                self._decode_graph_chunk_pages_lut,
                0,
                max_cache_pages.to(torch.int64).view(1),
            )
            kv_chunk_size = (decode_chunk_pages * self.page_size).to(torch.int32)
            max_chunks_per_req = int(self.request_indices.shape[0] // self.cache_seqlens.shape[0])
            update_regular_decode_graph_chunk_metadata(
                cache_seqlens=self.cache_seqlens,
                merge_indptr=self.merge_indptr,
                o_indptr=self.o_indptr,
                kv_chunk_size_ptr=self.kv_chunk_size_ptr,
                kv_chunk_size=kv_chunk_size,
                max_chunks_per_req=max_chunks_per_req,
            )
        else:
            from .graph_replay import update_decode_graph_chunk_metadata

            update_decode_graph_chunk_metadata(
                cache_seqlens=self.cache_seqlens,
                request_indices=self.request_indices,
                merge_indptr=self.merge_indptr,
                o_indptr=self.o_indptr,
                block_valid_mask=self.block_valid_mask,
                kv_chunk_size_ptr=self.kv_chunk_size_ptr,
                decode_chunk_pages_lut=self._decode_graph_chunk_pages_lut,
                page_size=self.page_size,
            )
        return self

    def prepare_for_cuda_graph_replay(
        self,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        *,
        fixed_split_size: int | None = None,
        disable_split_kv: bool = False,
    ) -> PagedAttentionWorkspace:
        if not self.use_cuda_graph:
            raise RuntimeError("prepare_for_cuda_graph_replay is only valid for graph-mode workspaces")
        return self.prepare(
            page_table,
            cache_seqlens,
            cu_seqlens_q,
            fixed_split_size=fixed_split_size,
            disable_split_kv=disable_split_kv,
        )

    def bind_cuda_graph_runtime_metadata(
        self,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
    ) -> PagedAttentionWorkspace:
        if not self.use_cuda_graph:
            raise RuntimeError("bind_cuda_graph_runtime_metadata is only valid for graph-mode workspaces")
        # Decode graph replay already selected the workspace by mode, so avoid
        # re-reading cu_seqlens_q back to the CPU just to rediscover q_len=1.
        inferred_mode = self.mode if self.mode == "decode" else infer_paged_mode(cu_seqlens_q)
        if inferred_mode != self.mode and not (
            self.mode == "extend" and inferred_mode == "decode"
        ) and not (
            self.mode == "verify" and inferred_mode == "extend"
        ):
            raise ValueError(f"workspace mode {self.mode} does not match bound mode {inferred_mode}")
        if page_table.device != self.device or cache_seqlens.device != self.device or cu_seqlens_q.device != self.device:
            raise ValueError("bound graph metadata must stay on the workspace device")
        self.page_table = page_table
        self.cache_seqlens = cache_seqlens
        self.cu_seqlens_q = cu_seqlens_q
        return self

    def prepare_decode_graph_replay_state(
        self,
        *,
        batch: int,
        total_q_capacity: int,
        max_page_table_width: int,
        max_cache_page_count: int,
    ) -> PagedAttentionWorkspace:
        if not self.use_cuda_graph:
            raise RuntimeError("prepare_decode_graph_replay_state is only valid for graph-mode workspaces")
        if self.mode != "decode":
            raise RuntimeError("prepare_decode_graph_replay_state is only valid for decode workspaces")
        if max_cache_page_count <= 0:
            raise ValueError("max_cache_page_count must be positive")

        from .graph_replay import make_decode_chunk_pages_lut_tensor, summarize_decode_chunk_pages_lut

        try:
            decode_chunk_pages_lut = build_decode_chunk_pages_lut(
                q_dtype=self.dtype,
                kv_dtype=self.kv_dtype,
                batch=batch,
                page_size=self.page_size,
                head_dim_qk=self.head_dim_qk,
                head_dim_vo=self.head_dim_vo,
                gqa_group_size=self.num_q_heads // self.num_kv_heads,
                max_effective_kv_pages=max_cache_page_count,
            )
        except KeyError:
            self._decode_graph_chunk_pages_lut = None
            self._decode_graph_max_chunks_per_req = None
            self._use_regular_decode_graph_replay = False
            self._decode_graph_metadata_captured_in_graph = False
            self.prepare_for_capacity(
                batch=batch,
                total_q_capacity=total_q_capacity,
                max_page_table_width=max_page_table_width,
                max_cache_seqlen=max_cache_page_count * self.page_size,
            )
            return self
        worst_page_count, max_chunks_per_req = summarize_decode_chunk_pages_lut(decode_chunk_pages_lut)
        self._decode_graph_chunk_pages_lut = make_decode_chunk_pages_lut_tensor(
            decode_chunk_pages_lut,
            device=self.device,
        )
        self._decode_graph_max_chunks_per_req = int(max_chunks_per_req)
        self._use_regular_decode_graph_replay = self.kv_dtype == torch.bfloat16 and batch >= 4
        self.prepare_for_capacity(
            batch=batch,
            total_q_capacity=total_q_capacity,
            max_page_table_width=max_page_table_width,
            max_cache_seqlen=worst_page_count * self.page_size,
        )
        self._validate_decode_graph_replay_capacity(batch=batch)
        return self

    def prepare_for_capacity(
        self,
        *,
        batch: int,
        total_q_capacity: int,
        max_page_table_width: int,
        max_cache_seqlen: int,
    ) -> PagedAttentionWorkspace:
        if batch <= 0:
            raise ValueError("batch must be positive")
        if total_q_capacity <= 0:
            raise ValueError("total_q_capacity must be positive")
        if max_page_table_width <= 0:
            raise ValueError("max_page_table_width must be positive")
        if max_cache_seqlen <= 0:
            raise ValueError("max_cache_seqlen must be positive")

        max_cache_seqlens = torch.full(
            (batch,),
            int(max_cache_seqlen),
            dtype=torch.int32,
            device=self.device,
        )
        num_cache_pages = int(self._plan_k_cache.shape[0]) if self._plan_k_cache is not None else 0
        if num_cache_pages <= 0:
            raise RuntimeError("paged workspace planning contract is not initialized")
        max_page_ids = torch.arange(max_page_table_width, dtype=torch.int32, device=self.device)
        max_page_table = (max_page_ids % num_cache_pages).unsqueeze(0).expand(batch, -1).contiguous()
        max_cu_seqlens_q = self._build_capacity_cu_seqlens_q(
            batch=batch,
            total_q_capacity=total_q_capacity,
        )
        return self.prepare(max_page_table, max_cache_seqlens, max_cu_seqlens_q)

    def _build_capacity_cu_seqlens_q(
        self,
        *,
        batch: int,
        total_q_capacity: int,
    ) -> torch.Tensor:
        if self.mode == "decode":
            return torch.arange(0, batch + 1, dtype=torch.int32, device=self.device)
        if total_q_capacity < batch:
            raise ValueError(
                f"extend graph bucket requires total_q_capacity >= batch, got {total_q_capacity} < {batch}"
            )
        q_lens = torch.ones(batch, dtype=torch.int32, device=self.device)
        q_lens[-1] = int(total_q_capacity - (batch - 1))
        cu_seqlens_q = torch.zeros(batch + 1, dtype=torch.int32, device=self.device)
        torch.cumsum(q_lens, dim=0, out=cu_seqlens_q[1:])
        return cu_seqlens_q

    def _ensure_plan_contract(self, active_total_q: int) -> None:
        if self._plan_q is None or self._plan_k_cache is None or self._plan_v_cache is None:
            raise RuntimeError("paged workspace planning contract is not initialized")
        if active_total_q <= int(self._plan_q.shape[0]):
            return
        if self.use_cuda_graph:
            raise ValueError(
                "graph-mode paged workspace capacity exceeded; construct a larger workspace or capture a larger graph bucket"
            )
        if self.fixed_capacity:
            raise ValueError(
                "fixed-capacity paged workspace exceeded; construct a larger eager extend workspace"
            )
        self._plan_q = _shape_only_cuda_tensor(
            (active_total_q, self.num_q_heads, self.head_dim_qk),
            dtype=self.dtype,
            device=self.device,
        )

    def update_decode_graph_replay_metadata(
        self,
        *,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
    ) -> PagedAttentionWorkspace:
        if not self.use_cuda_graph:
            raise RuntimeError("update_decode_graph_replay_metadata is only valid for graph-mode workspaces")
        if self.mode != "decode":
            raise RuntimeError("update_decode_graph_replay_metadata is only valid for decode workspaces")
        if self._decode_graph_chunk_pages_lut is None:
            raise RuntimeError("decode graph replay policy has not been prepared")
        if self.page_table is None:
            raise RuntimeError("decode graph workspace is missing page_table")
        if self.cache_seqlens is None:
            raise RuntimeError("decode graph workspace is missing cache_seqlens")
        if self.cu_seqlens_q is None:
            raise RuntimeError("decode graph workspace is missing cu_seqlens_q")
        if self.request_indices is None:
            raise RuntimeError("decode graph workspace is missing request indices")
        if self.block_valid_mask is None:
            raise RuntimeError("decode graph workspace is missing block_valid_mask")
        if self.merge_indptr is None or self.o_indptr is None:
            raise RuntimeError("decode graph workspace is missing indptr buffers")
        if self.kv_chunk_size_ptr is None:
            raise RuntimeError("decode graph workspace is missing kv_chunk_size_ptr")

        self._validate_decode_graph_replay_capacity(batch=int(self.cache_seqlens.shape[0]))

        if self._use_regular_decode_graph_replay:
            from .graph_replay import update_regular_decode_graph_replay_metadata

            update_regular_decode_graph_replay_metadata(
                req_to_token=req_to_token,
                req_pool_indices=req_pool_indices,
                page_table=self.page_table,
                cache_seqlens=self.cache_seqlens,
                merge_indptr=self.merge_indptr,
                o_indptr=self.o_indptr,
                kv_chunk_size_ptr=self.kv_chunk_size_ptr,
                decode_chunk_pages_lut=self._decode_graph_chunk_pages_lut,
                page_size=self.page_size,
            )
        else:
            from .graph_replay import update_decode_graph_replay_metadata

            update_decode_graph_replay_metadata(
                req_to_token=req_to_token,
                req_pool_indices=req_pool_indices,
                page_table=self.page_table,
                cache_seqlens=self.cache_seqlens,
                request_indices=self.request_indices,
                merge_indptr=self.merge_indptr,
                o_indptr=self.o_indptr,
                block_valid_mask=self.block_valid_mask,
                kv_chunk_size_ptr=self.kv_chunk_size_ptr,
                decode_chunk_pages_lut=self._decode_graph_chunk_pages_lut,
                page_size=self.page_size,
            )
        return self

    @torch._dynamo.disable
    def run(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        *,
        output: torch.Tensor,
        k_descale: torch.Tensor | None = None,
        v_descale: torch.Tensor | None = None,
        prepare_decode_graph_metadata: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from .api import paged_attention_forward

        if prepare_decode_graph_metadata is None:
            prepare_decode_graph_metadata = (
                self.use_cuda_graph
                and self.mode == "decode"
                and self._decode_graph_chunk_pages_lut is not None
                and self._plan is not None
                and torch.cuda.is_current_stream_capturing()
            )
        if prepare_decode_graph_metadata:
            if not self.use_cuda_graph:
                raise RuntimeError("prepare_decode_graph_metadata requires a graph-mode workspace")
            if self.mode != "decode":
                raise RuntimeError("prepare_decode_graph_metadata is only valid for decode workspaces")
            if self._decode_graph_chunk_pages_lut is None or self._plan is None:
                raise RuntimeError(
                    "prepare_decode_graph_metadata requires a decode replay-state workspace"
                )
            with record_function("paged_workspace.capture_decode_graph_replay_metadata"):
                self.update_decode_graph_replay_metadata_from_runtime_cache_seqlens()
            if torch.cuda.is_current_stream_capturing():
                self._decode_graph_metadata_captured_in_graph = True

        out, lse = paged_attention_forward(
            q,
            k_cache,
            v_cache,
            workspace=self,
            output=output,
            k_descale=k_descale,
            v_descale=v_descale,
        )
        return out, lse

    def current_lse_view(self) -> torch.Tensor:
        if self.lse is None:
            raise RuntimeError("workspace has not been prepared")
        return self.lse[:, : self.active_total_q].transpose(0, 1)

    def _validate_static_shapes(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> None:
        if q.device != self.device or k_cache.device != self.device or v_cache.device != self.device:
            raise ValueError("workspace inputs must stay on the workspace device")
        if q.dtype != self.dtype:
            raise TypeError(f"workspace expects q dtype {self.dtype}, got {q.dtype}")
        if k_cache.dtype != self.kv_dtype or v_cache.dtype != self.kv_dtype:
            raise TypeError(f"workspace expects kv dtype {self.kv_dtype}, got {k_cache.dtype}/{v_cache.dtype}")
        if tuple(q.shape[1:]) != (self.num_q_heads, self.head_dim_qk):
            raise ValueError(
                "q shape does not match the workspace contract: "
                f"expected (*, {self.num_q_heads}, {self.head_dim_qk}), got {tuple(q.shape)}"
            )
        if int(k_cache.shape[1]) != self.page_size or int(v_cache.shape[1]) != self.page_size:
            raise ValueError(f"workspace expects page_size={self.page_size}")
        if int(k_cache.shape[2]) != self.num_kv_heads or int(v_cache.shape[2]) != self.num_kv_heads:
            raise ValueError("kv head count does not match the workspace contract")
        if int(k_cache.shape[3]) != self.head_dim_qk:
            raise ValueError("k_cache head_dim does not match the workspace contract")
        if int(v_cache.shape[3]) != self.head_dim_vo:
            raise ValueError("v_cache head_dim does not match the workspace contract")

    def _ensure_capacity(self, plan: PagedPlan) -> None:
        work_items_needed = int(plan.new_batch_size)
        block_valid_needed = int(plan.padded_batch_size)
        total_q_needed = int(plan.total_q)
        batch_needed = int(plan.page_table_shape[0])
        page_table_width_needed = int(plan.page_table_shape[1])
        partial_rows_needed = int(plan.total_num_partial_rows) if plan.split_kv else 0

        work_items_capacity = 0 if self.request_indices is None else int(self.request_indices.shape[0])
        block_valid_capacity = 0 if self.block_valid_mask is None else int(self.block_valid_mask.shape[0])
        total_q_capacity = 0 if self.lse is None else int(self.lse.shape[1])
        batch_capacity = 0 if self.o_indptr is None else int(self.o_indptr.shape[0] - 1)
        page_table_width_capacity = 0 if self.page_table is None else int(self.page_table.shape[1])
        partial_rows_capacity = 0 if self.tmp_output is None else int(self.tmp_output.shape[0])

        needs_growth = (
            work_items_needed > work_items_capacity
            or block_valid_needed > block_valid_capacity
            or total_q_needed > total_q_capacity
            or batch_needed > batch_capacity
            or page_table_width_needed > page_table_width_capacity
            or partial_rows_needed > partial_rows_capacity
        )
        if not needs_growth:
            return
        if self.use_cuda_graph and self.request_indices is not None:
            raise ValueError(
                "graph-mode paged workspace capacity exceeded; construct a larger workspace or capture a larger graph bucket"
            )
        if self.fixed_capacity and self.request_indices is not None:
            raise ValueError(
                "fixed-capacity paged workspace exceeded; construct a larger eager extend workspace"
            )

        work_items_capacity = max(work_items_capacity, work_items_needed)
        block_valid_capacity = max(block_valid_capacity, block_valid_needed)
        total_q_capacity = max(total_q_capacity, total_q_needed)
        batch_capacity = max(batch_capacity, batch_needed)
        page_table_width_capacity = max(page_table_width_capacity, page_table_width_needed)
        partial_rows_capacity = max(partial_rows_capacity, partial_rows_needed)

        self._allocate_runtime_buffers(
            work_items_capacity=work_items_capacity,
            block_valid_capacity=block_valid_capacity,
            total_q_capacity=total_q_capacity,
            batch_capacity=batch_capacity,
            page_table_width_capacity=page_table_width_capacity,
            partial_rows_capacity=partial_rows_capacity,
        )

    def _allocate_runtime_buffers(
        self,
        *,
        work_items_capacity: int,
        block_valid_capacity: int,
        total_q_capacity: int,
        batch_capacity: int,
        page_table_width_capacity: int,
        partial_rows_capacity: int,
    ) -> None:
        self.request_indices = torch.empty(work_items_capacity, dtype=torch.int32, device=self.device)
        self.qo_tile_indices = torch.empty(work_items_capacity, dtype=torch.int32, device=self.device)
        self.kv_tile_indices = torch.empty(work_items_capacity, dtype=torch.int32, device=self.device)
        self.block_valid_mask = torch.empty(block_valid_capacity, dtype=torch.int32, device=self.device)
        self.page_table = torch.empty(
            (batch_capacity, page_table_width_capacity), dtype=torch.int32, device=self.device
        )
        self.cache_seqlens = torch.empty(batch_capacity, dtype=torch.int32, device=self.device)
        self.cu_seqlens_q = torch.empty(batch_capacity + 1, dtype=torch.int32, device=self.device)
        self.merge_indptr = torch.empty(total_q_capacity + 1, dtype=torch.int32, device=self.device)
        self.o_indptr = torch.empty(batch_capacity + 1, dtype=torch.int32, device=self.device)
        self.kv_chunk_size_ptr = torch.empty(1, dtype=torch.int32, device=self.device)
        self.total_num_rows_ptr = torch.empty(1, dtype=torch.int32, device=self.device)
        self.lse = torch.empty(
            _paged_lse_storage_shape(total_q_capacity, self.num_q_heads),
            dtype=torch.float32,
            device=self.device,
        )
        if partial_rows_capacity > 0:
            self.tmp_output = torch.empty(
                (partial_rows_capacity, self.num_q_heads, self.head_dim_vo),
                dtype=self.dtype,
                device=self.device,
            )
            self.tmp_lse = torch.empty(
                (partial_rows_capacity, self.num_q_heads),
                dtype=torch.float32,
                device=self.device,
            )
        else:
            self.tmp_output = None
            self.tmp_lse = None

    def _validate_decode_graph_replay_capacity(
        self,
        *,
        batch: int,
    ) -> None:
        if self._decode_graph_max_chunks_per_req is None:
            raise RuntimeError("decode graph replay policy has not been prepared")
        if self.request_indices is None:
            raise RuntimeError("decode graph workspace is missing request indices")
        if batch <= 0:
            raise ValueError("decode graph replay requires bs > 0")
        work_items_capacity = int(self.request_indices.shape[0])
        if work_items_capacity % batch != 0:
            raise RuntimeError(
                "decode graph workspace request_indices shape is incompatible with the batch bucket"
            )
        max_chunks_per_req = work_items_capacity // batch
        if max_chunks_per_req <= 0:
            raise RuntimeError("decode graph workspace must allocate at least one chunk per request")
        if self._decode_graph_max_chunks_per_req > max_chunks_per_req:
            raise RuntimeError(
                "decode graph workspace capacity is too small for the current chunking policy"
            )

    def _copy_runtime_metadata(
        self,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
    ) -> None:
        assert self.page_table is not None
        assert self.cache_seqlens is not None
        assert self.cu_seqlens_q is not None

        with record_function("paged_workspace.runtime_cast_metadata"):
            page_table_i32 = page_table if page_table.dtype == torch.int32 else page_table.to(torch.int32)
            cache_seqlens_i32 = (
                cache_seqlens if cache_seqlens.dtype == torch.int32 else cache_seqlens.to(torch.int32)
            )
            cu_seqlens_q_i32 = (
                cu_seqlens_q if cu_seqlens_q.dtype == torch.int32 else cu_seqlens_q.to(torch.int32)
            )

        with record_function("paged_workspace.runtime_copy_page_table"):
            self.page_table[: page_table_i32.shape[0], : page_table_i32.shape[1]].copy_(page_table_i32)
        with record_function("paged_workspace.runtime_copy_cache_seqlens"):
            self.cache_seqlens[: cache_seqlens_i32.shape[0]].copy_(cache_seqlens_i32)
        with record_function("paged_workspace.runtime_copy_cu_seqlens_q"):
            self.cu_seqlens_q[: cu_seqlens_q_i32.shape[0]].copy_(cu_seqlens_q_i32)

    def _copy_regular_decode_graph_plan_metadata(self, plan: PagedPlan) -> None:
        assert self.request_indices is not None
        assert self.merge_indptr is not None
        assert self.o_indptr is not None
        assert self.kv_chunk_size_ptr is not None
        assert self.total_num_rows_ptr is not None
        assert self.cache_seqlens is not None

        batch = int(plan.page_table_shape[0])
        if batch <= 0:
            raise RuntimeError("regular decode graph replay requires bs > 0")
        work_items_capacity = int(self.request_indices.shape[0])
        if work_items_capacity % batch != 0:
            raise RuntimeError(
                "decode graph workspace request_indices shape is incompatible with the batch bucket"
            )
        capture_max_chunks_per_req = work_items_capacity // batch
        current_work_items = len(plan.request_indices)
        if current_work_items % batch != 0:
            raise RuntimeError("decode graph replay plan work-items are incompatible with the batch bucket")
        current_max_chunks_per_req = current_work_items // batch
        if current_max_chunks_per_req > capture_max_chunks_per_req:
            raise RuntimeError("decode graph replay plan exceeds the captured fixed-grid capacity")

        from .graph_replay import update_regular_decode_graph_chunk_metadata

        update_regular_decode_graph_chunk_metadata(
            cache_seqlens=self.cache_seqlens,
            merge_indptr=self.merge_indptr,
            o_indptr=self.o_indptr,
            kv_chunk_size_ptr=self.kv_chunk_size_ptr,
            kv_chunk_size=int(plan.kv_chunk_size),
            max_chunks_per_req=capture_max_chunks_per_req,
        )
        self.total_num_rows_ptr[0] = int(plan.total_q)

    def _copy_plan_metadata(self, plan: PagedPlan) -> None:
        assert self.request_indices is not None
        assert self.qo_tile_indices is not None
        assert self.kv_tile_indices is not None
        assert self.merge_indptr is not None
        assert self.o_indptr is not None
        assert self.kv_chunk_size_ptr is not None
        assert self.total_num_rows_ptr is not None
        assert self.block_valid_mask is not None

        self._use_regular_decode_graph_replay = False
        if (
            self.kv_dtype == torch.bfloat16
            and int(plan.page_table_shape[0]) >= 4
            and self.use_cuda_graph
            and self._plan_has_regular_decode_graph_grid(plan)
        ):
            batch = int(plan.page_table_shape[0])
            work_items_capacity = int(self.request_indices.shape[0])
            if work_items_capacity % batch != 0:
                raise RuntimeError(
                    "decode graph workspace request_indices shape is incompatible with the batch bucket"
                )
            capture_max_chunks_per_req = work_items_capacity // batch
            current_max_chunks_per_req = len(plan.request_indices) // batch
            if self._decode_graph_max_chunks_per_req is None:
                self._decode_graph_max_chunks_per_req = int(capture_max_chunks_per_req)
            elif current_max_chunks_per_req > self._decode_graph_max_chunks_per_req:
                raise RuntimeError("decode graph replay plan exceeds the captured fixed-grid capacity")
            if current_max_chunks_per_req < capture_max_chunks_per_req:
                self._use_regular_decode_graph_replay = True
                self._copy_regular_decode_graph_plan_metadata(plan)
                return
            self._use_regular_decode_graph_replay = True

        with record_function("paged_workspace.plan_metadata_to_device"):
            request_indices = _copy_int_metadata(plan.request_indices, device=self.device)
            qo_tile_indices = _copy_int_metadata(plan.qo_tile_indices, device=self.device)
            kv_tile_indices = _copy_int_metadata(plan.kv_tile_indices, device=self.device)
            merge_indptr = _copy_int_metadata(plan.merge_indptr, device=self.device)
            o_indptr = _copy_int_metadata(plan.o_indptr, device=self.device)
            block_valid_mask = torch.tensor(plan.block_valid_mask, dtype=torch.int32, device=self.device)

        with record_function("paged_workspace.plan_metadata_zero_buffers"):
            self.request_indices.zero_()
            self.qo_tile_indices.zero_()
            self.kv_tile_indices.zero_()
            self.block_valid_mask.zero_()
        with record_function("paged_workspace.plan_metadata_copy_buffers"):
            self.request_indices[: request_indices.shape[0]].copy_(request_indices)
            self.qo_tile_indices[: qo_tile_indices.shape[0]].copy_(qo_tile_indices)
            self.kv_tile_indices[: kv_tile_indices.shape[0]].copy_(kv_tile_indices)
            self.merge_indptr[: merge_indptr.shape[0]].copy_(merge_indptr)
            self.o_indptr[: o_indptr.shape[0]].copy_(o_indptr)
            self.block_valid_mask[: block_valid_mask.shape[0]].copy_(block_valid_mask)
        with record_function("paged_workspace.plan_metadata_scalar_updates"):
            self.kv_chunk_size_ptr[0] = int(plan.kv_chunk_size)
            self.total_num_rows_ptr[0] = int(plan.total_q)
