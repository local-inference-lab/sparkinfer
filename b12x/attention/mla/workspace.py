"""Workspace state for sparse MLA execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from .split import default_sparse_mla_split_decode_config_for_width


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
    """Return a tiny CUDA tensor whose shape/stride/dtype/device are stable.

    Used as a phantom in host-launcher cache keys so that varying batch sizes
    do not trigger CUTLASS recompilation.  The tensor is never read by kernels.
    """
    base = torch.empty(1, dtype=dtype, device=device)
    return base.as_strided(shape, (0,) * len(shape))


@dataclass(kw_only=True)
class MLAWorkspace:
    mode: Literal["decode", "extend", "verify", "draft_extend"]
    device: torch.device
    dtype: torch.dtype
    kv_dtype: torch.dtype
    num_q_heads: int
    head_dim: int
    v_head_dim: int
    topk: int
    max_total_q: int
    max_batch: int
    page_size: int = 64
    padded_heads: int = 128
    use_cuda_graph: bool = False
    fixed_capacity: bool = False
    max_chunks_per_row: int = 64
    page_table_1: torch.Tensor | None = None
    cache_seqlens_int32: torch.Tensor | None = None
    nsa_cache_seqlens_int32: torch.Tensor | None = None
    page_table_1_runtime: torch.Tensor | None = None
    cache_seqlens_int32_runtime: torch.Tensor | None = None
    nsa_cache_seqlens_int32_runtime: torch.Tensor | None = None
    tmp_output: torch.Tensor | None = None
    tmp_lse: torch.Tensor | None = None
    kv_chunk_size_ptr: torch.Tensor | None = None
    num_chunks_ptr: torch.Tensor | None = None
    sm_scale_tensor: torch.Tensor | None = None
    sm_scale_value: float | None = None
    kv_chunk_size_value: int | None = None
    num_chunks_value: int | None = None
    # Phantom tensors for stable host-launcher cache keys (fixed_capacity only).
    _contract_q: torch.Tensor | None = None
    _contract_page_table: torch.Tensor | None = None
    _contract_nsa_cache_seqlens: torch.Tensor | None = None
    _contract_output: torch.Tensor | None = None
    _contract_tmp_output: torch.Tensor | None = None
    _contract_tmp_lse: torch.Tensor | None = None

    @classmethod
    def for_contract(
        cls,
        *,
        mode: Literal["decode", "extend", "verify", "draft_extend"],
        device: torch.device | str,
        dtype: torch.dtype,
        kv_dtype: torch.dtype,
        num_q_heads: int,
        head_dim: int,
        v_head_dim: int,
        topk: int,
        max_total_q: int,
        max_batch: int,
        page_size: int = 64,
        use_cuda_graph: bool = False,
        padded_heads: int = 128,
    ) -> MLAWorkspace:
        device = _canonical_device(device)
        workspace = cls(
            mode=mode,
            device=device,
            dtype=dtype,
            kv_dtype=kv_dtype,
            num_q_heads=num_q_heads,
            head_dim=head_dim,
            v_head_dim=v_head_dim,
            topk=topk,
            max_total_q=int(max_total_q),
            max_batch=int(max_batch),
            page_size=page_size,
            padded_heads=padded_heads,
            use_cuda_graph=use_cuda_graph,
        )
        workspace._allocate_split_buffers()
        if use_cuda_graph:
            workspace._allocate_runtime_metadata()
        return workspace

    @classmethod
    def for_fixed_capacity(
        cls,
        *,
        mode: Literal["decode", "extend", "verify", "draft_extend"],
        device: torch.device | str,
        dtype: torch.dtype,
        kv_dtype: torch.dtype,
        num_q_heads: int,
        head_dim: int,
        v_head_dim: int,
        topk: int,
        max_total_q: int,
        max_batch: int,
        page_size: int = 64,
        use_cuda_graph: bool = False,
        padded_heads: int = 128,
    ) -> MLAWorkspace:
        workspace = cls.for_contract(
            mode=mode,
            device=device,
            dtype=dtype,
            kv_dtype=kv_dtype,
            num_q_heads=num_q_heads,
            head_dim=head_dim,
            v_head_dim=v_head_dim,
            topk=topk,
            max_total_q=max_total_q,
            max_batch=max_batch,
            page_size=page_size,
            use_cuda_graph=use_cuda_graph,
            padded_heads=padded_heads,
        )
        workspace.fixed_capacity = True
        workspace._initialize_split_chunk_config_if_needed()
        workspace._allocate_contract_phantoms()
        if use_cuda_graph:
            workspace._allocate_runtime_metadata()
        return workspace

    def _allocate_runtime_metadata(self) -> None:
        if self.page_table_1_runtime is None:
            self.page_table_1_runtime = torch.empty(
                (self.max_total_q, self.topk),
                dtype=torch.int32,
                device=self.device,
            )
        if self.cache_seqlens_int32_runtime is None:
            self.cache_seqlens_int32_runtime = torch.empty(
                (self.max_batch,),
                dtype=torch.int32,
                device=self.device,
            )
        if self.nsa_cache_seqlens_int32_runtime is None:
            self.nsa_cache_seqlens_int32_runtime = torch.empty(
                (self.max_total_q,),
                dtype=torch.int32,
                device=self.device,
            )

    def _allocate_split_buffers(self) -> None:
        if self.mode not in ("decode", "extend", "verify", "draft_extend"):
            return
        if self.tmp_output is None:
            self.tmp_output = torch.empty(
                (self.max_total_q, self.num_q_heads, self.max_chunks_per_row, self.v_head_dim),
                dtype=self.dtype,
                device=self.device,
            )
        if self.tmp_lse is None:
            self.tmp_lse = torch.empty(
                (self.max_total_q, self.num_q_heads, self.max_chunks_per_row),
                dtype=torch.float32,
                device=self.device,
            )
        if self.kv_chunk_size_ptr is None:
            self.kv_chunk_size_ptr = torch.empty((1,), dtype=torch.int32, device=self.device)
            self.kv_chunk_size_value = None
        if self.num_chunks_ptr is None:
            self.num_chunks_ptr = torch.empty((1,), dtype=torch.int32, device=self.device)
            self.num_chunks_value = None

    def _initialize_split_chunk_config_if_needed(self) -> None:
        self._allocate_split_buffers()
        if not (self.fixed_capacity or self.use_cuda_graph):
            return
        if self.kv_chunk_size_value is not None and self.num_chunks_value is not None:
            return
        split_cfg = default_sparse_mla_split_decode_config_for_width(int(self.topk))
        if split_cfg is None:
            return
        assert self.kv_chunk_size_ptr is not None
        assert self.num_chunks_ptr is not None
        self.kv_chunk_size_ptr[0] = int(split_cfg.chunk_size)
        self.num_chunks_ptr[0] = int(split_cfg.num_chunks)
        self.kv_chunk_size_value = int(split_cfg.chunk_size)
        self.num_chunks_value = int(split_cfg.num_chunks)

    def set_split_chunk_config(self, *, kv_chunk_size: int, num_chunks: int) -> None:
        if num_chunks <= 0 or num_chunks > self.max_chunks_per_row:
            raise ValueError(
                f"num_chunks must be in [1, {self.max_chunks_per_row}], got {num_chunks}"
            )
        if kv_chunk_size <= 0:
            raise ValueError(f"kv_chunk_size must be positive, got {kv_chunk_size}")
        self._allocate_split_buffers()
        assert self.kv_chunk_size_ptr is not None
        assert self.num_chunks_ptr is not None
        if self.kv_chunk_size_value != int(kv_chunk_size):
            self.kv_chunk_size_ptr.fill_(int(kv_chunk_size))
            self.kv_chunk_size_value = int(kv_chunk_size)
        if self.num_chunks_value != int(num_chunks):
            self.num_chunks_ptr.fill_(int(num_chunks))
            self.num_chunks_value = int(num_chunks)

    def set_decode_chunk_config(self, *, kv_chunk_size: int, num_chunks: int) -> None:
        self.set_split_chunk_config(kv_chunk_size=kv_chunk_size, num_chunks=num_chunks)

    def prepare_decode(
        self,
        page_table_1: torch.Tensor,
        cache_seqlens_int32: torch.Tensor,
        nsa_cache_seqlens_int32: torch.Tensor,
    ) -> None:
        self._prepare_sparse(
            page_table_1=page_table_1,
            cache_seqlens_int32=cache_seqlens_int32,
            nsa_cache_seqlens_int32=nsa_cache_seqlens_int32,
        )

    def prepare_extend(
        self,
        selected_token_offsets: torch.Tensor,
        cache_seqlens_int32: torch.Tensor,
        nsa_cache_seqlens_int32: torch.Tensor,
    ) -> None:
        self._prepare_sparse(
            page_table_1=selected_token_offsets,
            cache_seqlens_int32=cache_seqlens_int32,
            nsa_cache_seqlens_int32=nsa_cache_seqlens_int32,
        )

    def bind_cuda_graph_runtime_metadata(
        self,
        *,
        page_table_1: torch.Tensor,
        cache_seqlens_int32: torch.Tensor,
        nsa_cache_seqlens_int32: torch.Tensor,
    ) -> None:
        if not self.use_cuda_graph:
            raise RuntimeError("bind_cuda_graph_runtime_metadata is only valid for graph-mode workspaces")
        self._prepare_sparse(
            page_table_1=page_table_1,
            cache_seqlens_int32=cache_seqlens_int32,
            nsa_cache_seqlens_int32=nsa_cache_seqlens_int32,
        )

    def _prepare_sparse(
        self,
        *,
        page_table_1: torch.Tensor,
        cache_seqlens_int32: torch.Tensor,
        nsa_cache_seqlens_int32: torch.Tensor,
    ) -> None:
        if page_table_1.ndim != 2:
            raise ValueError(f"page_table_1 must be rank-2, got {tuple(page_table_1.shape)}")
        if cache_seqlens_int32.ndim != 1:
            raise ValueError(
                f"cache_seqlens_int32 must be rank-1, got {tuple(cache_seqlens_int32.shape)}"
            )
        if nsa_cache_seqlens_int32.ndim != 1:
            raise ValueError(
                "nsa_cache_seqlens_int32 must be rank-1, "
                f"got {tuple(nsa_cache_seqlens_int32.shape)}"
            )
        if page_table_1.device != self.device:
            raise ValueError(
                f"page_table_1 device {page_table_1.device} does not match workspace device {self.device}"
            )
        if cache_seqlens_int32.device != self.device:
            raise ValueError(
                "cache_seqlens_int32 device "
                f"{cache_seqlens_int32.device} does not match workspace device {self.device}"
            )
        if nsa_cache_seqlens_int32.device != self.device:
            raise ValueError(
                "nsa_cache_seqlens_int32 device "
                f"{nsa_cache_seqlens_int32.device} does not match workspace device {self.device}"
            )
        if page_table_1.dtype != torch.int32:
            raise ValueError(f"page_table_1 must have dtype torch.int32, got {page_table_1.dtype}")
        if cache_seqlens_int32.dtype != torch.int32:
            raise ValueError(
                "cache_seqlens_int32 must have dtype torch.int32, "
                f"got {cache_seqlens_int32.dtype}"
            )
        if nsa_cache_seqlens_int32.dtype != torch.int32:
            raise ValueError(
                "nsa_cache_seqlens_int32 must have dtype torch.int32, "
                f"got {nsa_cache_seqlens_int32.dtype}"
            )
        if page_table_1.shape[0] > self.max_total_q:
            raise ValueError(
                f"page_table_1 rows {page_table_1.shape[0]} exceed workspace capacity {self.max_total_q}"
            )
        if cache_seqlens_int32.shape[0] > self.max_batch:
            raise ValueError(
                "cache_seqlens_int32 batch "
                f"{cache_seqlens_int32.shape[0]} exceeds workspace capacity {self.max_batch}"
            )
        if page_table_1.shape[1] > self.topk:
            raise ValueError(
                f"page_table_1 width {page_table_1.shape[1]} exceeds topk capacity {self.topk}"
            )
        if page_table_1.shape[0] != nsa_cache_seqlens_int32.shape[0]:
            raise ValueError(
                "page_table_1 rows "
                f"{page_table_1.shape[0]} do not match nsa_cache_seqlens_int32 rows "
                f"{nsa_cache_seqlens_int32.shape[0]}"
            )
        use_runtime_buffers = self.use_cuda_graph
        if not use_runtime_buffers:
            if self.device.type == "cuda":
                self.page_table_1 = page_table_1
                self.cache_seqlens_int32 = cache_seqlens_int32
                self.nsa_cache_seqlens_int32 = nsa_cache_seqlens_int32
            else:
                self.page_table_1 = page_table_1.clone()
                self.cache_seqlens_int32 = cache_seqlens_int32.clone()
                self.nsa_cache_seqlens_int32 = nsa_cache_seqlens_int32.clone()
            return

        self._allocate_runtime_metadata()
        assert self.page_table_1_runtime is not None
        assert self.cache_seqlens_int32_runtime is not None
        assert self.nsa_cache_seqlens_int32_runtime is not None
        rows, width = page_table_1.shape
        batch = cache_seqlens_int32.shape[0]
        self.page_table_1_runtime[:rows, :width].copy_(page_table_1)
        self.cache_seqlens_int32_runtime[:batch].copy_(cache_seqlens_int32)
        self.nsa_cache_seqlens_int32_runtime[:rows].copy_(nsa_cache_seqlens_int32)
        self.page_table_1 = self.page_table_1_runtime[:rows, :width]
        self.cache_seqlens_int32 = self.cache_seqlens_int32_runtime[:batch]
        self.nsa_cache_seqlens_int32 = self.nsa_cache_seqlens_int32_runtime[:rows]

    def _allocate_contract_phantoms(self) -> None:
        """Create zero-stride phantom tensors at max capacity for stable cache keys."""
        # q is viewed as uint32 in the kernel: (max_total_q, num_q_heads, head_dim // 4).
        self._contract_q = _shape_only_cuda_tensor(
            (self.max_total_q, self.num_q_heads, self.head_dim // 4),
            dtype=torch.uint32,
            device=self.device,
        )
        self._contract_page_table = _shape_only_cuda_tensor(
            (self.max_total_q, self.topk),
            dtype=torch.int32,
            device=self.device,
        )
        self._contract_nsa_cache_seqlens = _shape_only_cuda_tensor(
            (self.max_total_q,),
            dtype=torch.int32,
            device=self.device,
        )
        self._contract_output = _shape_only_cuda_tensor(
            (self.max_total_q, self.num_q_heads, self.v_head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        if self.tmp_output is not None and self.tmp_lse is not None:
            self._contract_tmp_output = _shape_only_cuda_tensor(
                (self.max_total_q, self.num_q_heads, self.max_chunks_per_row, self.v_head_dim),
                dtype=self.dtype,
                device=self.device,
            )
            self._contract_tmp_lse = _shape_only_cuda_tensor(
                (self.max_total_q, self.num_q_heads, self.max_chunks_per_row),
                dtype=torch.float32,
                device=self.device,
            )

