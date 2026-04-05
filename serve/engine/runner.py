"""Model runner — executes forward passes through the transformer stack.

Manages workspace pools, pre-allocates static buffers for CUDA graph
capture, and provides prefill/decode entry points.
"""

from __future__ import annotations

import json
import os

import torch
import torch.nn.functional as F

from b12x.profiling import record_function

from serve.cache.kv_cache import KVCacheManager
from serve.cache.page_pool import PagePool
from serve.logging import StartupSession, get_logger
from serve.model.attention import B12xPagedAttention
from serve.model.loader import LoadedModel
from serve.model.ops import rms_norm
from serve.model.workspaces import get_b12x_moe_workspace_pool

LOGGER = get_logger(__name__)


def _shared_attn_workspace_key(
    attn: B12xPagedAttention,
    *,
    kv_dtype: torch.dtype,
    page_size: int,
    num_cache_pages: int,
    max_total_q: int,
    max_batch: int,
    max_page_table_width: int,
    use_cuda_graph: bool,
) -> tuple[object, ...]:
    return attn.workspace_cache_key(
        kv_dtype=kv_dtype,
        page_size=page_size,
        num_cache_pages=num_cache_pages,
        max_total_q=max_total_q,
        max_batch=max_batch,
        max_page_table_width=max_page_table_width,
        use_cuda_graph=use_cuda_graph,
    )

class ModelRunner:
    """Runs forward passes through the full model stack."""

    def __init__(
        self,
        model: LoadedModel,
        kv_mgr=None,
        device: torch.device | str = "cuda",
        max_batch_size: int = 128,
        max_total_tokens: int = 4096,
        pool: PagePool | None = None,
        ssm_pool=None,
    ):
        self.model = model
        self.kv_mgr = kv_mgr
        self.pool = pool or (kv_mgr.pool if kv_mgr is not None else None)
        self.ssm_pool = ssm_pool
        self.device = torch.device(device)
        self.cfg = model.config
        self.max_batch_size = max_batch_size
        self.max_total_tokens = max_total_tokens

        # Inject per-layer workspaces and bind per-layer cache refs.
        moe_workspace = get_b12x_moe_workspace_pool(self.device)
        shared_attn_workspaces: dict[tuple[object, ...], dict[str, object]] = {}

        kv_layer_idx = 0
        ssm_layer_idx = 0
        layer_types = getattr(model.config, 'layer_types', None)

        for i, layer in enumerate(model.layers):
            attn = getattr(layer, 'attn', None)
            lt = layer_types[i] if layer_types else None
            if isinstance(attn, B12xPagedAttention):
                workspace_key = _shared_attn_workspace_key(
                    attn,
                    kv_dtype=self.pool.k_cache[kv_layer_idx].dtype,
                    page_size=self.pool.page_size,
                    num_cache_pages=self.pool.num_pages,
                    max_total_q=self.max_total_tokens,
                    max_batch=self.max_batch_size,
                    max_page_table_width=self.pool.num_pages,
                    use_cuda_graph=False,
                )
                workspace = shared_attn_workspaces.get(workspace_key)
                if workspace is None:
                    workspace = attn.allocate_workspaces(
                        device=self.device,
                        kv_dtype=self.pool.k_cache[kv_layer_idx].dtype,
                        page_size=self.pool.page_size,
                        num_cache_pages=self.pool.num_pages,
                        max_total_q=self.max_total_tokens,
                        max_batch=self.max_batch_size,
                        max_page_table_width=self.pool.num_pages,
                        use_cuda_graph=False,
                    )
                    shared_attn_workspaces[workspace_key] = workspace
                attn.set_workspace(workspace)
            layer.set_moe_workspace(moe_workspace)

            # Bind cache refs so layers own their slice.
            if lt == "linear_attention" and ssm_pool is not None:
                layer.bind_cache(
                    ssm_state=ssm_pool.ssm_state_for_layer(ssm_layer_idx),
                    conv_state=ssm_pool.conv_state_for_layer(ssm_layer_idx),
                )
                ssm_layer_idx += 1
            else:
                layer.bind_cache(
                    k_cache=self.pool.k_cache[kv_layer_idx],
                    v_cache=self.pool.v_cache[kv_layer_idx],
                )
                kv_layer_idx += 1

        # CUDA graph pool (lazily populated).
        self._graph_pool = None
        self._compiled_layers = None
        self._workspace_refresh_needed = False

    def prefill(
        self,
        token_ids: torch.Tensor,
        request_ids: list[int],
        q_seqlens: list[int],
    ) -> torch.Tensor:
        """Run prefill for one or more requests. Returns next-token logits.

        token_ids: [total_tokens] int64, concatenated prompt tokens.
        request_ids: which requests these tokens belong to.
        q_seqlens: number of tokens per request.
        """
        # KV management: either scheduler does it (kv_mgr=None) or we do.
        if self.kv_mgr is not None:
            for rid, qlen in zip(request_ids, q_seqlens):
                if rid not in self.kv_mgr._requests:
                    self.kv_mgr.allocate_request(rid)
                self.kv_mgr.extend_request(rid, qlen)

        return self._forward(token_ids, request_ids, q_seqlens, mode="extend")

    def compile_model(self, mode: str = "max-autotune-no-cudagraphs") -> None:
        """Compile per-layer wrappers for all modes.

        b12x kernel calls within each layer are wrapped with
        @torch.compiler.disable so dynamo doesn't trace into them.
        """
        self._compiled_layers = [
            torch.compile(layer, mode=mode)
            for layer in self.model.layers
        ]

    def _refresh_paged_attention_workspaces(self) -> None:
        """Recreate paged-attention workspaces after compiled prefill/extend.

        The compiled prefill path can leave the next eager decode unstable for
        one-page KV pools. Rebinding fresh workspaces before decode avoids
        reusing that poisoned runtime metadata.
        """
        if self.pool is None:
            return

        kv_layer_idx = 0
        shared_attn_workspaces: dict[tuple[object, ...], dict[str, object]] = {}
        for layer in self.model.layers:
            attn = getattr(layer, "attn", None)
            if not isinstance(attn, B12xPagedAttention):
                continue
            workspace_key = _shared_attn_workspace_key(
                attn,
                kv_dtype=self.pool.k_cache[kv_layer_idx].dtype,
                page_size=self.pool.page_size,
                num_cache_pages=self.pool.num_pages,
                max_total_q=self.max_total_tokens,
                max_batch=self.max_batch_size,
                max_page_table_width=self.pool.num_pages,
                use_cuda_graph=False,
            )
            workspace = shared_attn_workspaces.get(workspace_key)
            if workspace is None:
                workspace = attn.allocate_workspaces(
                    device=self.device,
                    kv_dtype=self.pool.k_cache[kv_layer_idx].dtype,
                    page_size=self.pool.page_size,
                    num_cache_pages=self.pool.num_pages,
                    max_total_q=self.max_total_tokens,
                    max_batch=self.max_batch_size,
                    max_page_table_width=self.pool.num_pages,
                    use_cuda_graph=False,
                )
                shared_attn_workspaces[workspace_key] = workspace
            attn.set_workspace(workspace)
            kv_layer_idx += 1

    def capture_decode_graphs(
        self,
        batch_sizes: list[int] | None = None,
        prefill_chunk_size: int | None = None,
        startup: StartupSession | None = None,
    ) -> None:
        """Capture CUDA graphs for decode and optionally prefill."""
        if batch_sizes is None:
            batch_sizes = [1, 2, 4, 8]
        from serve.engine.cuda_graph import GraphPool
        total_graphs = len(batch_sizes) + (1 if prefill_chunk_size else 0)
        if startup is not None:
            startup.start("Capture CUDA graphs", total=total_graphs)
        self._graph_pool = GraphPool(
            self.model, self.pool, self.device,
            batch_sizes=batch_sizes,
            prefill_chunk_size=prefill_chunk_size,
            progress_callback=(
                None
                if startup is None
                else lambda desc: startup.advance(description=desc)
            ),
        )
        if startup is not None:
            startup.finish()

    def warmup(
        self,
        batch_sizes: list[int] | None = None,
        prefill_lengths: list[int] | None = None,
        warm_decode: bool = True,
        startup: StartupSession | None = None,
    ) -> None:
        """Pre-compile kernels for common shapes.

        Runs dummy prefill and optionally decode at each
        (batch_size, prefill_length) combination to trigger CuTe DSL kernel
        compilation upfront.
        """
        if batch_sizes is None:
            batch_sizes = [1]
        if prefill_lengths is None:
            # Cover common prompt lengths: short, medium (chat template), long.
            prefill_lengths = [4, 64, 256]

        import time
        t0 = time.time()
        total_steps = 0
        for plen in prefill_lengths:
            decode_steps = 1 if warm_decode else 0
            if self._warmup_shape_fits(batch_size=1, prefill_len=plen, decode_steps=decode_steps):
                total_steps += 1 + int(warm_decode)
        for bs in batch_sizes:
            if bs == 1:
                continue
            decode_steps = 2 if warm_decode else 0
            if self._warmup_shape_fits(batch_size=bs, prefill_len=4, decode_steps=decode_steps):
                total_steps += bs + (2 if warm_decode else 0)
        if startup is not None:
            startup.start("Warmup kernels", total=total_steps)

        for plen in prefill_lengths:
            decode_steps = 1 if warm_decode else 0
            if not self._warmup_shape_fits(batch_size=1, prefill_len=plen, decode_steps=decode_steps):
                continue
            # Prefill at various lengths to compile attention for those shapes.
            if startup is not None:
                startup.advance(description=f"Warmup kernels [dim](prefill {plen} tok)[/]")
            dummy_ids = torch.ones(plen, dtype=torch.long, device=self.device)
            self.prefill(dummy_ids, request_ids=[10000], q_seqlens=[plen])
            if warm_decode:
                if startup is not None:
                    startup.advance(description=f"Warmup kernels [dim](decode after {plen} tok)[/]")
                # One decode step.
                dummy_tok = torch.ones(1, dtype=torch.long, device=self.device)
                self.decode(dummy_tok, request_ids=[10000])
            self.kv_mgr.free_request(10000)

        for bs in batch_sizes:
            if bs == 1:
                continue  # Already covered above.
            decode_steps = 2 if warm_decode else 0
            if not self._warmup_shape_fits(batch_size=bs, prefill_len=4, decode_steps=decode_steps):
                continue
            for i in range(bs):
                if startup is not None:
                    startup.advance(description=f"Warmup kernels [dim](prefill bs={bs}, req={i + 1}/{bs})[/]")
                dummy_ids = torch.ones(4, dtype=torch.long, device=self.device)
                self.prefill(dummy_ids, request_ids=[10000 + i], q_seqlens=[4])
            rids = [10000 + i for i in range(bs)]
            if warm_decode:
                for _ in range(2):
                    if startup is not None:
                        startup.advance(description=f"Warmup kernels [dim](decode bs={bs})[/]")
                    dummy_tok = torch.ones(bs, dtype=torch.long, device=self.device)
                    self.decode(dummy_tok, request_ids=rids)
            for i in range(bs):
                self.kv_mgr.free_request(10000 + i)

        torch.cuda.synchronize()
        if startup is not None:
            startup.finish()
        LOGGER.info(f"Warmup complete ({time.time() - t0:.1f}s)")

    def _warmup_shape_fits(self, *, batch_size: int, prefill_len: int, decode_steps: int) -> bool:
        """Return whether a dummy warmup shape fits the currently bound KV pool."""
        if self.pool is None:
            return True
        total_len_per_request = prefill_len + decode_steps
        pages_per_request = (total_len_per_request + self.pool.page_size - 1) // self.pool.page_size
        return batch_size * pages_per_request <= self.pool.num_pages

    def decode(
        self,
        token_ids: torch.Tensor,
        request_ids: list[int],
    ) -> torch.Tensor:
        """Run one decode step for all active requests. Returns next-token logits.

        token_ids: [batch] int64, one token per request.
        """
        # Allocate 1 new KV slot per request.
        with record_function("runner.decode"):
            for rid in request_ids:
                self.kv_mgr.extend_request(rid, 1)

            bs = len(request_ids)
            if self._graph_pool is not None:
                graph = self._graph_pool.get(bs)
                if graph is not None:
                    return self._replay_graph(graph, token_ids, request_ids)

            q_seqlens = [1] * bs
            return self._forward(token_ids, request_ids, q_seqlens, mode="decode")

    def _replay_graph(self, graph, token_ids, request_ids):
        """Replay a captured CUDA graph."""
        with record_function("runner.replay_graph"):
            device = self.device
            page_table = self.kv_mgr.build_page_table(request_ids, device=device)
            cache_seqlens = self.kv_mgr.build_cache_seqlens(request_ids, device=device)
            pre_write = cache_seqlens - 1
            positions = pre_write.long()
            return graph.replay(token_ids, page_table, cache_seqlens, pre_write, positions)

    def forward_batch(
        self,
        token_ids: torch.Tensor,
        q_seqlens: list[int] | torch.Tensor,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        mode: str,
        graph_bs: int | None = None,
        ssm_cache_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with pre-built batch tensors (scheduler-driven)."""
        with record_function(f"runner.forward_batch.{mode}"):
            device = self.device
            q_lens_t = _as_q_seqlens_tensor(q_seqlens, device)
            bs = cache_seqlens.shape[0]

            if graph_bs is not None and self._graph_pool is not None:
                if mode == "decode":
                    graph = self._graph_pool.get(graph_bs)
                else:
                    graph = self._graph_pool.get_prefill(graph_bs)
                if graph is not None:
                    with record_function("runner.graph_lookup_hit"):
                        pass
                    with record_function("runner.graph_build_pre_write"):
                        pre_write = cache_seqlens - q_lens_t
                    with record_function("runner.graph_build_positions"):
                        positions = _build_positions(q_lens_t, pre_write, device)
                    with record_function("runner.graph_replay_call"):
                        return graph.replay(
                            token_ids,
                            page_table,
                            cache_seqlens,
                            pre_write,
                            positions,
                            ssm_cache_indices=ssm_cache_indices,
                        )

            # Build cu_seqlens_q only for eager path.
            cu_seqlens_q = torch.zeros(int(q_lens_t.shape[0]) + 1, dtype=torch.int32, device=device)
            cu_seqlens_q[1:] = q_lens_t.cumsum(0)
            q_seqlens_list = _q_seqlens_list(q_seqlens)

            return self._forward_inner(token_ids, q_seqlens_list, page_table, cache_seqlens,
                                       cu_seqlens_q, mode, ssm_cache_indices)

    def _forward(
        self,
        token_ids: torch.Tensor,
        request_ids: list[int],
        q_seqlens: list[int],
        mode: str,
    ) -> torch.Tensor:
        """Forward pass with KVCacheManager-built tensors (warmup/legacy)."""
        with record_function(f"runner.forward.{mode}"):
            cfg = self.cfg
            device = self.device

            page_table = self.kv_mgr.build_page_table(request_ids, device=device)
            cache_seqlens = self.kv_mgr.build_cache_seqlens(request_ids, device=device)
            cu_seqlens_q = self.kv_mgr.build_cu_seqlens_q(q_seqlens, device=device)
            return self._forward_inner(token_ids, q_seqlens, page_table, cache_seqlens, cu_seqlens_q, mode)

    def _forward_inner(self, token_ids, q_seqlens, page_table, cache_seqlens,
                        cu_seqlens_q, mode, ssm_cache_indices=None):
        """Core forward pass — unified for all model types."""
        with record_function(f"runner.forward_inner.{mode}"):
            from serve.engine.step_state import StepState
            cfg = self.cfg
            device = self.device

            pre_write_seqlens = cache_seqlens - torch.tensor(
                q_seqlens, dtype=torch.int32, device=device
            )
            positions = _build_positions(q_seqlens, pre_write_seqlens, device)
            hidden = self.model.embed_tokens(token_ids.to(device))
            layer_trace_path = os.environ.get("B12X_LAYER_STDS_PATH")
            layer_stds = None
            if layer_trace_path and mode != "decode":
                layer_stds = [float(hidden.float().std().item())]

            # Build MambaForwardMetadata if SSM indices are provided.
            mamba_meta = None
            if ssm_cache_indices is not None:
                from serve.engine.mamba_metadata import MambaForwardMetadata
                is_decode = (mode == "decode")
                mamba_meta = MambaForwardMetadata(
                    cache_indices=ssm_cache_indices,
                    has_initial_states=torch.ones(ssm_cache_indices.shape[0], dtype=torch.bool, device=device)
                        if is_decode
                        else (pre_write_seqlens[:ssm_cache_indices.shape[0]] > 0),
                    cu_seqlens=cu_seqlens_q if not is_decode else None,
                    seq_lens=q_seqlens if not is_decode else None,
                )

            state = StepState(
                cos=self.model.cos,
                sin=self.model.sin,
                positions=positions,
                page_table=page_table,
                cache_seqlens=pre_write_seqlens,
                cu_seqlens_q=cu_seqlens_q,
                mamba=mamba_meta,
                is_decode=(mode == "decode"),
                attn_workspaces_prepared=False,
                prepared_attn_workspace_ids=set(),
            )

            if mode == "decode" and self._workspace_refresh_needed:
                self._refresh_paged_attention_workspaces()
                self._workspace_refresh_needed = False

            layers = self._compiled_layers if self._compiled_layers is not None else self.model.layers

            for layer_idx, layer in enumerate(layers):
                with record_function(f"runner.layer.{layer_idx}.{mode}"):
                    hidden = layer(hidden, state)
                if layer_stds is not None:
                    layer_stds.append(float(hidden.float().std().item()))

            if self._compiled_layers is not None:
                self._workspace_refresh_needed = True

            with record_function("runner.final_norm"):
                hidden = rms_norm(hidden, self.model.final_norm_weight, cfg.rms_norm_eps,
                                  gemma_style=getattr(cfg, 'gemma_norm', False))
            with record_function("runner.lm_head"):
                logits = F.linear(hidden, self.model.lm_head_weight)

            if layer_stds is not None:
                try:
                    import torch.distributed as dist
                    if (not dist.is_initialized()) or dist.get_rank() == 0:
                        with open(layer_trace_path, "w") as f:
                            json.dump(layer_stds, f)
                except Exception:
                    pass

            # Return only the last token's logits per request.
            if all(s == 1 for s in q_seqlens):
                return logits  # Decode: one token per request, all are "last".
            cu = cu_seqlens_q[1:] - 1  # Last index per request.
            return logits[cu.long()]


def _as_q_seqlens_tensor(
    q_seqlens: list[int] | torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(q_seqlens, torch.Tensor):
        if q_seqlens.device != device:
            return q_seqlens.to(device=device, dtype=torch.int32)
        return q_seqlens if q_seqlens.dtype == torch.int32 else q_seqlens.to(torch.int32)
    return torch.tensor(q_seqlens, device=device, dtype=torch.int32)


def _q_seqlens_list(q_seqlens: list[int] | torch.Tensor) -> list[int]:
    if isinstance(q_seqlens, torch.Tensor):
        return [int(v) for v in q_seqlens.tolist()]
    return q_seqlens


def _build_positions(
    q_seqlens: list[int] | torch.Tensor,
    pre_write_seqlens: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Build position tensor for RoPE.

    For each request, positions are [cache_len, cache_len+1, ..., cache_len+q_len-1].
    """
    q_lens_t = _as_q_seqlens_tensor(q_seqlens, device)
    total_q = int(q_lens_t.sum().item())
    if int(q_lens_t.shape[0]) == 1:
        # Single request — simple arange. Graph-safe.
        return (pre_write_seqlens[0] + torch.arange(total_q, device=device)).long()

    batch_ids = torch.repeat_interleave(
        torch.arange(int(q_lens_t.shape[0]), device=device), q_lens_t.long()
    )
    cu = torch.zeros(int(q_lens_t.shape[0]) + 1, device=device, dtype=torch.int32)
    cu[1:] = q_lens_t.cumsum(0)
    offsets = torch.arange(total_q, device=device) - cu[batch_ids]
    return (pre_write_seqlens[batch_ids] + offsets).long()
