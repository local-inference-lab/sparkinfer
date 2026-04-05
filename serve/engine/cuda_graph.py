"""CUDA graph capture and replay for decode and chunked prefill.

Each CapturedGraph owns the full state for one (batch_size, tokens_per_req)
shape: static input buffers, output buffers, workspace pools, and the
captured graph itself.

- Decode graphs: tokens_per_req=1, one token per request.
- Prefill graphs: tokens_per_req=chunk_size, one chunk per request.

The GraphPool holds decode and prefill graphs keyed by batch size.
"""

from __future__ import annotations

from typing import Callable, Optional

import os
import torch
import torch.nn.functional as F

from b12x.profiling import record_function

from serve.engine.runner import _shared_attn_workspace_key
from serve.engine.step_state import StepState
from serve.model.attention import B12xPagedAttention
from serve.model.ops import rms_norm
from serve.model.workspaces import get_b12x_moe_workspace_pool


def _graph_debug_enabled() -> bool:
    return os.environ.get("B12X_SERVE_GRAPH_DEBUG", "").strip().lower() in {"1", "true", "yes"}


def _graph_debug(message: str) -> None:
    if _graph_debug_enabled():
        print(f"[serve-graph] {message}", flush=True)


class CapturedGraph:
    """One captured CUDA graph for a specific (bs, tokens_per_req) shape."""

    def __init__(
        self,
        model,
        pool,
        bs: int,
        device: torch.device,
        tokens_per_req: int = 1,
    ):
        self.bs = bs
        self.tokens_per_req = tokens_per_req
        self.device = device
        self.model = model
        self.pool = pool
        cfg = model.config

        total_q = bs * tokens_per_req

        # Static input buffers.
        self.token_ids = torch.zeros(total_q, dtype=torch.long, device=device)
        self.page_table = torch.zeros(bs, pool.num_pages, dtype=torch.int32, device=device)
        self.cache_seqlens = torch.ones(bs, dtype=torch.int32, device=device) * (tokens_per_req + 1)
        self.pre_write = torch.ones(bs, dtype=torch.int32, device=device)
        self.cu_seqlens_q = torch.arange(
            0, total_q + 1, tokens_per_req, dtype=torch.int32, device=device
        )
        self.positions = torch.zeros(total_q, dtype=torch.long, device=device)

        # Mamba metadata (hybrid models only).
        self._has_mamba = False
        self.mamba_cache_indices = None
        if getattr(cfg, 'layer_types', None) and any(t == "linear_attention" for t in cfg.layer_types):
            self._has_mamba = True
            self.mamba_cache_indices = torch.zeros(bs, dtype=torch.int64, device=device)

        # Static output buffer — only last token per request.
        self.output = torch.empty(bs, cfg.vocab_size, dtype=torch.bfloat16, device=device)

        # Per-graph workspaces.
        self._attn_workspaces = []
        shared_attn_workspaces: dict[tuple[object, ...], dict[str, object]] = {}
        self._moe_workspace = get_b12x_moe_workspace_pool(device)

        # Per-layer output buffers for b12x kernels.
        layer_types = getattr(cfg, 'layer_types', None)
        self._attn_outputs = []
        kv_layer_idx = 0
        for i, layer in enumerate(self.model.layers):
            lt = layer_types[i] if layer_types else "attention"
            attn = getattr(layer, "attn", None)
            if lt == "attention" or lt is None:
                self._attn_outputs.append(
                    torch.empty(total_q, cfg.num_q_heads, cfg.head_dim, dtype=torch.bfloat16, device=device))
                if isinstance(attn, B12xPagedAttention):
                    workspace_key = _shared_attn_workspace_key(
                        attn,
                        kv_dtype=pool.k_cache[kv_layer_idx].dtype,
                        page_size=pool.page_size,
                        num_cache_pages=pool.num_pages,
                        max_total_q=total_q,
                        max_batch=bs,
                        max_page_table_width=pool.num_pages,
                        use_cuda_graph=True,
                    )
                    workspace = shared_attn_workspaces.get(workspace_key)
                    if workspace is None:
                        workspace = attn.allocate_workspaces(
                            device=device,
                            kv_dtype=pool.k_cache[kv_layer_idx].dtype,
                            page_size=pool.page_size,
                            num_cache_pages=pool.num_pages,
                            max_total_q=total_q,
                            max_batch=bs,
                            max_page_table_width=pool.num_pages,
                            use_cuda_graph=True,
                        )
                        self._prime_attn_workspace(workspace)
                        shared_attn_workspaces[workspace_key] = workspace
                    self._attn_workspaces.append(workspace)
                else:
                    self._attn_workspaces.append(None)
                kv_layer_idx += 1
            else:
                self._attn_outputs.append(None)
                self._attn_workspaces.append(None)
        self._moe_outputs = [
            torch.empty(total_q, cfg.hidden_size, dtype=torch.bfloat16, device=device)
            for _ in range(cfg.num_layers)
        ]

        self.graph: Optional[torch.cuda.CUDAGraph] = None

    def _prime_attn_workspace(self, workspace_set) -> None:
        """Prime graph workspaces with the same capacity APIs used by sglang."""
        if workspace_set is None:
            return
        decode_workspace = workspace_set["decode"]
        if decode_workspace.mode == "decode":
            decode_workspace.prepare_decode_graph_replay_state(
                batch=self.bs,
                total_q_capacity=self.bs,
                max_page_table_width=self.pool.num_pages,
                max_cache_page_count=self.pool.num_pages,
            )
        else:
            decode_workspace.prepare_for_capacity(
                batch=self.bs,
                total_q_capacity=self.bs,
                max_page_table_width=self.pool.num_pages,
                max_cache_seqlen=self.pool.num_pages * self.pool.page_size,
            )
        workspace_set["extend"].prepare_for_capacity(
            batch=self.bs,
            total_q_capacity=self.bs * self.tokens_per_req,
            max_page_table_width=self.pool.num_pages,
            max_cache_seqlen=self.pool.num_pages * self.pool.page_size,
        )

    def capture(self) -> None:
        """Warmup and capture the graph."""
        _graph_debug(f"capture:start bs={self.bs} tokens_per_req={self.tokens_per_req}")
        self._inject_buffers()
        _graph_debug("capture:inject_buffers:done")
        self._prepare_attn_workspaces()
        _graph_debug("capture:prepare_attn_workspaces:done")

        capture_stream = torch.cuda.Stream()
        with torch.cuda.stream(capture_stream):
            for warmup_iter in range(3):
                _graph_debug(f"capture:warmup_iter:{warmup_iter}:start")
                self.output.copy_(self._run_forward())
                _graph_debug(f"capture:warmup_iter:{warmup_iter}:forward_done")
            torch.cuda.current_stream().synchronize()
            _graph_debug("capture:warmup_sync:done")

        self.graph = torch.cuda.CUDAGraph()
        _graph_debug("capture:graph_enter")
        with torch.cuda.graph(self.graph, stream=capture_stream):
            self.output.copy_(self._run_forward())
        _graph_debug("capture:graph_exit")

        self._restore_buffers()
        _graph_debug("capture:restore_buffers:done")

    def replay(
        self,
        token_ids: torch.Tensor,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        pre_write: torch.Tensor,
        positions: torch.Tensor,
        ssm_cache_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Update inputs and replay."""
        bs = cache_seqlens.shape[0]
        total_q = token_ids.shape[0]
        with record_function("graph.replay_update_inputs"):
            with record_function("graph.copy_token_ids"):
                self.token_ids[:total_q].copy_(token_ids)
            with record_function("graph.copy_page_table"):
                self.page_table[:bs, :page_table.shape[1]].copy_(page_table)
            with record_function("graph.copy_cache_seqlens"):
                self.cache_seqlens[:bs].copy_(cache_seqlens)
            with record_function("graph.copy_pre_write"):
                self.pre_write[:bs].copy_(pre_write)
            with record_function("graph.copy_positions"):
                self.positions[:total_q].copy_(positions)
            if ssm_cache_indices is not None and self.mamba_cache_indices is not None:
                with record_function("graph.copy_ssm_cache_indices"):
                    self.mamba_cache_indices[:bs].copy_(ssm_cache_indices)

        with record_function("graph.prepare_attn_workspaces"):
            self._prepare_attn_workspaces()
        with record_function("graph.cuda_graph_replay"):
            self.graph.replay()
        return self.output[:bs]

    def _run_forward(self) -> torch.Tensor:
        """Forward pass using static buffers. Works for both decode and prefill."""
        cfg = self.model.config
        _graph_debug("run_forward:embed:start")
        hidden = self.model.embed_tokens(self.token_ids)
        _graph_debug("run_forward:embed:done")

        mamba_meta = None
        if self._has_mamba and self.mamba_cache_indices is not None:
            from serve.engine.mamba_metadata import MambaForwardMetadata
            is_decode = (self.tokens_per_req == 1)
            mamba_meta = MambaForwardMetadata(
                cache_indices=self.mamba_cache_indices,
                has_initial_states=torch.ones(self.bs, dtype=torch.bool, device=self.device)
                    if is_decode
                    else (self.pre_write > 0),
                cu_seqlens=self.cu_seqlens_q if not is_decode else None,
                seq_lens=[self.tokens_per_req] * self.bs if not is_decode else None,
            )

        state = StepState(
            cos=self.model.cos,
            sin=self.model.sin,
            positions=self.positions,
            page_table=self.page_table,
            cache_seqlens=self.pre_write,
            cu_seqlens_q=self.cu_seqlens_q,
            mamba=mamba_meta,
            is_decode=(self.tokens_per_req == 1),
            attn_workspaces_prepared=True,
            prepared_attn_workspace_ids=None,
        )

        for layer_idx, layer in enumerate(self.model.layers):
            if layer_idx in (0, len(self.model.layers) // 2, len(self.model.layers) - 1):
                _graph_debug(f"run_forward:layer:{layer_idx}:start")
            hidden = layer(hidden, state)
            if layer_idx in (0, len(self.model.layers) // 2, len(self.model.layers) - 1):
                _graph_debug(f"run_forward:layer:{layer_idx}:done")

        _graph_debug("run_forward:final_norm:start")
        hidden = rms_norm(hidden, self.model.final_norm_weight, cfg.rms_norm_eps,
                          gemma_style=getattr(cfg, 'gemma_norm', False))
        _graph_debug("run_forward:final_norm:done")
        _graph_debug("run_forward:lm_head:start")
        logits = F.linear(hidden, self.model.lm_head_weight)
        _graph_debug("run_forward:lm_head:done")

        # Extract last token per request.
        if self.tokens_per_req == 1:
            return logits
        last_indices = torch.arange(
            self.tokens_per_req - 1, self.bs * self.tokens_per_req, self.tokens_per_req,
            dtype=torch.long, device=self.device,
        )
        return logits[last_indices]

    def _inject_buffers(self) -> None:
        """Inject per-graph workspaces and output buffers into layers."""
        self._saved_attn_ws = []
        self._saved_moe_ws = []
        self._saved_attn_out = []
        self._saved_moe_out = []

        for i, layer in enumerate(self.model.layers):
            attn = getattr(layer, 'attn', None)
            is_paged = isinstance(attn, B12xPagedAttention)
            self._saved_attn_ws.append(getattr(attn, '_workspace', None) if is_paged else None)
            self._saved_moe_ws.append(getattr(layer, '_moe_workspace', getattr(layer.ffn, '_moe_workspace', None)))
            self._saved_attn_out.append(getattr(attn, '_output_buffer', None) if is_paged else None)
            self._saved_moe_out.append(getattr(layer.ffn, '_moe_output_buffer', None))

            if is_paged:
                attn.set_workspace(self._attn_workspaces[i])
                if self._attn_outputs[i] is not None:
                    attn.set_output_buffer(self._attn_outputs[i])
            layer.set_moe_workspace(self._moe_workspace)
            layer.set_moe_output_buffer(self._moe_outputs[i])

    def _prepare_attn_workspaces(self) -> None:
        mode = "decode" if self.tokens_per_req == 1 else "extend"
        seen: set[int] = set()
        for workspace_set in self._attn_workspaces:
            if workspace_set is None:
                continue
            workspace_set_id = id(workspace_set)
            if workspace_set_id in seen:
                continue
            seen.add(workspace_set_id)
            workspace = workspace_set[mode]
            with record_function(f"graph.workspace_prepare.{mode}"):
                workspace.prepare_for_cuda_graph_replay(
                    self.page_table,
                    self.cache_seqlens,
                    self.cu_seqlens_q,
                )

    def _restore_buffers(self) -> None:
        """Restore shared workspaces after capture."""
        for i, layer in enumerate(self.model.layers):
            attn = getattr(layer, 'attn', None)
            is_paged = isinstance(attn, B12xPagedAttention)
            if is_paged and self._saved_attn_ws[i] is not None:
                attn.set_workspace(self._saved_attn_ws[i])
            if self._saved_moe_ws[i] is not None:
                layer.set_moe_workspace(self._saved_moe_ws[i])
            if is_paged:
                attn._output_buffer = self._saved_attn_out[i]
            layer.set_moe_output_buffer(self._saved_moe_out[i])


class GraphPool:
    """Pool of captured graphs for decode and prefill."""

    def __init__(
        self,
        model,
        pool,
        device,
        batch_sizes: list[int] | None = None,
        prefill_chunk_size: int | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ):
        self.model = model
        self.pool = pool
        self.device = device
        self._progress_callback = progress_callback
        self._decode_graphs: dict[int, CapturedGraph] = {}
        self._prefill_graphs: dict[int, CapturedGraph] = {}

        if batch_sizes:
            self.capture_decode(batch_sizes)
        if prefill_chunk_size:
            self.capture_prefill(prefill_chunk_size)

    def capture_decode(self, batch_sizes: list[int]) -> None:
        """Capture decode graphs (1 token per request)."""
        for bs in batch_sizes:
            if bs not in self._decode_graphs:
                if self._progress_callback is not None:
                    self._progress_callback(f"Capture CUDA graphs [dim](decode bs={bs})[/]")
                g = CapturedGraph(self.model, self.pool, bs, self.device, tokens_per_req=1)
                g.capture()
                self._decode_graphs[bs] = g

    def capture_prefill(self, chunk_size: int, batch_sizes: list[int] | None = None) -> None:
        """Capture prefill graphs (chunk_size tokens per request)."""
        for bs in (batch_sizes or [1]):
            if bs not in self._prefill_graphs:
                if self._progress_callback is not None:
                    self._progress_callback(
                        f"Capture CUDA graphs [dim](prefill bs={bs}, chunk={chunk_size})[/]"
                    )
                g = CapturedGraph(self.model, self.pool, bs, self.device, tokens_per_req=chunk_size)
                g.capture()
                self._prefill_graphs[bs] = g

    def get(self, bs: int) -> Optional[CapturedGraph]:
        """Get decode graph for exact batch size, or None."""
        return self._decode_graphs.get(bs)

    def get_prefill(self, bs: int) -> Optional[CapturedGraph]:
        """Get prefill graph for exact batch size, or None."""
        return self._prefill_graphs.get(bs)

    @property
    def captured_sizes(self) -> list[int]:
        return sorted(self._decode_graphs.keys())
