"""Top-level serving engine with TP coordination.

The ServingEngine is the single abstraction that both CLI and API use.
It handles model loading, warmup, TP broadcast, chat templates, and
generation using the BatchScheduler + PrefixCheckpointCache.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist

from transformers import AutoTokenizer

from b12x.profiling import record_function
from serve.cache.page_pool import PagePool
from serve.cache.prefix_checkpoint_cache import PrefixCheckpointCache
from serve.engine.request import Request
from serve.engine.sampling import SamplingParams, sample_batch
from serve.engine.scheduler import BatchScheduler
from serve.logging import get_logger, start_startup_session
from serve.model.loader import load_model, LoadedModel
from serve.tp.launch import get_active_tp_health
from serve.tp.group import TPGroup

LOGGER = get_logger(__name__)


def _startup_debug_enabled() -> bool:
    return os.environ.get("B12X_SERVE_STARTUP_DEBUG", "").strip().lower() in {"1", "true", "yes"}


def _startup_debug(message: str, *, rank: int) -> None:
    if rank == 0 and _startup_debug_enabled():
        print(f"[serve-startup] {message}", flush=True)

@dataclass
class GenerationResult:
    """Public result returned to callers."""
    request_id: int
    prompt_ids: list[int]
    generated_ids: list[int]
    finished: bool = False
    finish_reason: str | None = None
    time_to_first_token_ms: float = 0.0
    total_time_ms: float = 0.0


def _should_enable_prefix_cache(
    *,
    is_hybrid: bool,
    has_state_snapshot_slots: bool,
) -> bool:
    """Decide whether prefix-cache reuse is safe for the current engine mode."""
    if not is_hybrid:
        return True
    return has_state_snapshot_slots


def _should_enable_layer_compile(
    *,
    is_hybrid: bool,
    compile_layers: bool,
) -> bool:
    """Decide whether per-layer torch.compile should run."""
    del is_hybrid
    return compile_layers


def _estimate_loaded_model_bytes(model: LoadedModel) -> int:
    """Return the persistent GPU storage owned by *model*.

    Walks the LoadedModel object graph and counts unique CUDA storages so
    tied/shared tensors are only charged once. Many of the custom runtime
    containers in `serve.model` and `b12x` hold tensors in plain attributes
    or dataclass fields rather than registered parameters/buffers.
    """

    seen_storages: set[tuple[int, int]] = set()
    seen_objects: set[int] = set()
    total = 0

    def visit(obj) -> None:
        nonlocal total
        if obj is None:
            return
        if isinstance(obj, (str, bytes, int, float, bool, torch.dtype, torch.device)):
            return

        if isinstance(obj, torch.Tensor):
            if obj.device.type != "cuda":
                return
            storage = obj.untyped_storage()
            key = (storage.data_ptr(), storage.nbytes())
            if key in seen_storages:
                return
            seen_storages.add(key)
            total += storage.nbytes()
            return

        obj_id = id(obj)
        if obj_id in seen_objects:
            return
        seen_objects.add(obj_id)

        if isinstance(obj, dict):
            for key, value in obj.items():
                visit(key)
                visit(value)
            return

        if isinstance(obj, (list, tuple, set, frozenset)):
            for item in obj:
                visit(item)
            return

        if hasattr(obj, "__dict__"):
            for value in vars(obj).values():
                visit(value)

    visit(model)
    return total


def _format_gib(num_bytes: int) -> str:
    return f"{num_bytes / (1024 ** 3):.2f} GiB"


class ServingEngine:
    """Unified serving engine for all ranks."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        tp_group: Optional[TPGroup] = None,
        load_backend: str = "auto",
        kv_dtype: torch.dtype = torch.bfloat16,
        warmup_prefill_lengths: list[int] | None = None,
        graph_batch_sizes: list[int] | None = None,
        prefill_chunk_size: int = 512,
        capture_prefill_graph: bool = False,
        compile_layers: bool = True,
        max_running: int | None = None,
        max_waiting: int = 256,
        max_prefill_tokens: int = 4096,
    ):
        torch.set_grad_enabled(False)
        self.tp_group = tp_group
        self.rank = tp_group.rank if tp_group else 0
        self.world_size = tp_group.world_size if tp_group else 1
        self.device = device
        self.model_path = model_path
        self._next_rid = 0
        self._lock = threading.Lock()  # Guards scheduler + _next_rid.
        self._loop_running = False
        self._work_event = threading.Event()  # Wakes the server loop on submit.
        self._loop_error: str | None = None

        if self.rank == 0:
            LOGGER.info(f"Loading model [bold](TP={self.world_size})[/]")
        _startup_debug("load_model:start", rank=self.rank)

        self.model = load_model(
            model_path,
            device=device,
            tp_group=tp_group,
            load_backend=load_backend,
        )
        _startup_debug("load_model:done", rank=self.rank)
        self.cfg = self.model.config
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
        _startup_debug("tokenizer:done", rank=self.rank)
        if not self.tokenizer.chat_template:
            import pathlib
            jinja_path = pathlib.Path(model_path) / "chat_template.jinja"
            if jinja_path.exists():
                self.tokenizer.chat_template = jinja_path.read_text()

        # KV cache — only for self-attention layers.
        layer_types = getattr(self.cfg, 'layer_types', None)
        num_kv_layers = self.cfg.num_layers
        if layer_types:
            num_kv_layers = sum(1 for t in layer_types if t == "attention")

        free_mem, total_mem = torch.cuda.mem_get_info()
        model_bytes = _estimate_loaded_model_bytes(self.model)
        # KV cache + model weights together should occupy mem_fraction of total GPU memory.
        mem_fraction = 0.75
        target_mem = int(total_mem * mem_fraction)
        kv_budget = max(0, target_mem - model_bytes)
        kv_budget_before_tp = kv_budget

        model_bytes_min = model_bytes
        model_bytes_max = model_bytes
        free_mem_min = free_mem
        free_mem_max = free_mem

        # All ranks must agree on num_pages for TP broadcast shapes.
        if self.world_size > 1:
            model_bytes_min_t = torch.tensor([model_bytes], dtype=torch.long, device=device)
            model_bytes_max_t = torch.tensor([model_bytes], dtype=torch.long, device=device)
            free_mem_min_t = torch.tensor([free_mem], dtype=torch.long, device=device)
            free_mem_max_t = torch.tensor([free_mem], dtype=torch.long, device=device)
            budget_t = torch.tensor([kv_budget], dtype=torch.long, device=device)
            dist.all_reduce(model_bytes_min_t, op=dist.ReduceOp.MIN)
            dist.all_reduce(model_bytes_max_t, op=dist.ReduceOp.MAX)
            dist.all_reduce(free_mem_min_t, op=dist.ReduceOp.MIN)
            dist.all_reduce(free_mem_max_t, op=dist.ReduceOp.MAX)
            dist.all_reduce(budget_t, op=dist.ReduceOp.MIN)
            model_bytes_min = model_bytes_min_t.item()
            model_bytes_max = model_bytes_max_t.item()
            free_mem_min = free_mem_min_t.item()
            free_mem_max = free_mem_max_t.item()
            kv_budget = budget_t.item()

        # Linear-attention state arena shared by live requests and cached
        # terminal checkpoints.
        linear_state_arena = None
        live_ssm_slots = 0
        graph_sizes = [1, 2, 4, 8] if graph_batch_sizes is None else graph_batch_sizes
        max_running = max_running or (max(graph_sizes) if graph_sizes else 8)
        if layer_types and any(t == "linear_attention" for t in layer_types):
            from serve.cache.linear_state_arena import LinearStateArena
            from serve.cache.tensor_arena import TensorArena

            num_linear_layers = sum(1 for t in layer_types if t == "linear_attention")
            live_ssm_slots = max_running
            num_heads = self.cfg.linear_num_v_heads
            head_v_dim = self.cfg.linear_head_v_dim
            head_k_dim = self.cfg.linear_head_k_dim
            conv_dim = (
                self.cfg.linear_num_k_heads * self.cfg.linear_head_k_dim * 2
                + self.cfg.linear_num_v_heads * self.cfg.linear_head_v_dim
            )
            conv_kernel = self.cfg.linear_conv_kernel
            live_ssm_bytes = TensorArena.estimate_memory_bytes(
                num_slots=live_ssm_slots,
                num_linear_layers=num_linear_layers,
                num_heads=num_heads,
                head_v_dim=head_v_dim,
                head_k_dim=head_k_dim,
                conv_dim=conv_dim,
                conv_kernel=conv_kernel,
            )
            kv_budget = max(0, kv_budget - live_ssm_bytes)

            slot_bytes = TensorArena.slot_memory_bytes_for_shape(
                num_linear_layers=num_linear_layers,
                num_heads=num_heads,
                head_v_dim=head_v_dim,
                head_k_dim=head_k_dim,
                conv_dim=conv_dim,
                conv_kernel=conv_kernel,
            )
            max_cached_snapshot_slots = 64
            snapshot_slots = 0
            snapshot_budget = kv_budget // 8
            if slot_bytes > 0 and snapshot_budget >= 2 * slot_bytes:
                snapshot_slots = min(
                    max_cached_snapshot_slots,
                    max(0, snapshot_budget // slot_bytes - 1),
                )

            linear_state_arena = LinearStateArena(
                live_slots=live_ssm_slots,
                snapshot_slots=snapshot_slots,
                num_linear_layers=num_linear_layers,
                num_heads=num_heads,
                head_v_dim=head_v_dim,
                head_k_dim=head_k_dim,
                conv_dim=conv_dim,
                conv_kernel=conv_kernel,
                device=device,
            )
            snapshot_ssm_bytes = (
                TensorArena.estimate_memory_bytes(
                    num_slots=snapshot_slots,
                    num_linear_layers=num_linear_layers,
                    num_heads=num_heads,
                    head_v_dim=head_v_dim,
                    head_k_dim=head_k_dim,
                    conv_dim=conv_dim,
                    conv_kernel=conv_kernel,
                )
                if snapshot_slots > 0
                else 0
            )
            kv_budget = max(0, kv_budget - snapshot_ssm_bytes)
            if self.rank == 0:
                LOGGER.info(
                    f"LinearStateArena: {live_ssm_slots} live + "
                    f"{snapshot_slots} checkpoint slots, "
                    f"{linear_state_arena.memory_bytes() / 1e6:.0f} MB"
                )

        kv_bytes_per_token_block = (
            64
            * self.cfg.num_kv_heads
            * self.cfg.head_dim
            * (torch.finfo(kv_dtype).bits // 8 if kv_dtype.is_floating_point else 1)
            * 2
            * num_kv_layers
        )
        num_pages = PagePool.estimate_num_pages(
            kv_budget,
            num_layers=num_kv_layers,
            kv_heads=self.cfg.num_kv_heads,
            head_dim=self.cfg.head_dim,
            kv_dtype=kv_dtype,
        )
        kv_pool_bytes = num_pages * kv_bytes_per_token_block

        if self.rank == 0:
            LOGGER.debug(
                "KV sizing: "
                f"target={_format_gib(target_mem)} "
                f"total={_format_gib(total_mem)} "
                f"free_after_load[min,max]=[{_format_gib(free_mem_min)}, {_format_gib(free_mem_max)}] "
                f"model_bytes[min,max]=[{_format_gib(model_bytes_min)}, {_format_gib(model_bytes_max)}] "
                f"kv_budget_before_tp={_format_gib(kv_budget_before_tp)} "
                f"kv_budget={_format_gib(kv_budget)}"
            )
            LOGGER.debug(
                "KV layout: "
                f"layers={num_kv_layers} "
                f"kv_heads={self.cfg.num_kv_heads} "
                f"head_dim={self.cfg.head_dim} "
                f"dtype={kv_dtype} "
                f"bytes_per_page={_format_gib(kv_bytes_per_token_block)} "
                f"num_pages={num_pages} "
                f"pool_bytes={_format_gib(kv_pool_bytes)} "
                f"cuda_allocated={_format_gib(torch.cuda.memory_allocated())} "
                f"cuda_reserved={_format_gib(torch.cuda.memory_reserved())}"
            )

        self.pool = PagePool(
            num_pages=num_pages,
            num_layers=num_kv_layers,
            kv_heads=self.cfg.num_kv_heads,
            head_dim=self.cfg.head_dim,
            kv_dtype=kv_dtype,
            device=device,
        )
        _startup_debug("page_pool:done", rank=self.rank)
        self.cache = PrefixCheckpointCache(
            self.pool,
            state_arena=linear_state_arena,
        )

        # Model runner.
        from serve.engine.runner import ModelRunner
        from serve.cache.kv_cache import KVCacheManager
        warmup_kv_mgr = KVCacheManager(self.pool)
        runner_max_total_tokens = max(max_prefill_tokens, prefill_chunk_size)
        self.runner = ModelRunner(
            self.model,
            warmup_kv_mgr,
            device=device,
            max_batch_size=max_running,
            max_total_tokens=runner_max_total_tokens,
            pool=self.pool,
            ssm_pool=linear_state_arena,
        )
        _startup_debug("runner:done", rank=self.rank)

        if self.rank == 0:
            LOGGER.info(f"KV cache: {self.pool.num_pages} pages ({self.pool.num_pages * 64} tokens)")

        is_hybrid = layer_types is not None and any(t == "linear_attention" for t in layer_types)

        enable_layer_compile = _should_enable_layer_compile(
            is_hybrid=is_hybrid,
            compile_layers=compile_layers,
        )
        warm_decode = not bool(graph_sizes)
        startup = start_startup_session()
        if not is_hybrid:
            # Warmup uses kv_mgr-based path which doesn't support SSM layers.
            if self.rank == 0:
                LOGGER.info("Warming up")
            _startup_debug("warmup:start", rank=self.rank)
            prefill_lengths = warmup_prefill_lengths or [4, 64]
            self.runner.warmup(
                batch_sizes=[1],
                prefill_lengths=prefill_lengths,
                warm_decode=warm_decode,
                startup=startup,
            )
            _startup_debug("warmup:done", rank=self.rank)

        if enable_layer_compile:
            if self.rank == 0:
                LOGGER.info("Compiling layers")
            _startup_debug("compile:start", rank=self.rank)
            self.runner.compile_model()
            _startup_debug("compile:done", rank=self.rank)

        if enable_layer_compile:
            if is_hybrid:
                if self.rank == 0:
                    LOGGER.info("Skipping legacy post-compile warmup for hybrid model; first real request will materialize compiled layers")
            else:
                _startup_debug("post_compile_warmup:start", rank=self.rank)
                self.runner.warmup(
                    batch_sizes=[1],
                    prefill_lengths=[4, prefill_chunk_size],
                    warm_decode=warm_decode,
                    startup=startup,
                )
                _startup_debug("post_compile_warmup:done", rank=self.rank)

        if graph_sizes:
            if self.rank == 0:
                LOGGER.info("Capturing CUDA graphs")
            _startup_debug(f"capture_decode_graphs:start batch_sizes={graph_sizes}", rank=self.rank)
            self.runner.capture_decode_graphs(
                batch_sizes=graph_sizes,
                prefill_chunk_size=prefill_chunk_size if capture_prefill_graph else None,
                startup=startup,
            )
            _startup_debug("capture_decode_graphs:done", rank=self.rank)

        # Warmup/compile/graph capture all touch shared runtime state. Reset it
        # before the scheduler starts serving real requests.
        if linear_state_arena is not None:
            linear_state_arena.zero_all()
        for i in range(len(self.pool.k_cache)):
            self.pool.k_cache[i].zero_()
            self.pool.v_cache[i].zero_()

        # Scheduler.
        hybrid_prefix_cache = _should_enable_prefix_cache(
            is_hybrid=is_hybrid,
            has_state_snapshot_slots=(
                linear_state_arena is not None
                and linear_state_arena.num_snapshot_slots > 0
            ),
        )

        self.scheduler = BatchScheduler(
            cache=self.cache,
            pool=self.pool,
            ssm_pool=linear_state_arena,
            enable_prefix_cache=hybrid_prefix_cache,
            captured_bs=graph_sizes,
            max_running=max_running,
            max_waiting=max_waiting,
            max_prefill_tokens=max_prefill_tokens,
            chunk_size=prefill_chunk_size,
            device=device,
        )
        self._runtime_policy = {
            "graph_batch_sizes": list(graph_sizes),
            "prefill_chunk_size": prefill_chunk_size,
            "capture_prefill_graph": capture_prefill_graph,
            "compile_layers": enable_layer_compile,
            "max_running": max_running,
            "max_waiting": max_waiting,
            "max_prefill_tokens": max_prefill_tokens,
            "prefix_cache_enabled": hybrid_prefix_cache,
        }
        self._tp_health: dict | None = None

        if self.world_size > 1:
            _startup_debug("dist_barrier:start", rank=self.rank)
            barrier_kwargs = {}
            if self.device.startswith("cuda"):
                barrier_kwargs["device_ids"] = [torch.device(self.device).index]
            dist.barrier(**barrier_kwargs)
            _startup_debug("dist_barrier:done", rank=self.rank)

        if self.rank == 0:
            LOGGER.info("Ready")
        _startup_debug("ready", rank=self.rank)

    # -- public API (rank 0) -----------------------------------------------

    def submit(
        self,
        input_ids: list[int],
        params: SamplingParams | None = None,
        timeout_s: float | None = None,
    ) -> Request:
        """Submit a request without blocking. Returns the Request object.

        Does NOT broadcast to TP followers — use generate() or
        generate_batch() for TP-safe generation. submit() is intended
        for single-GPU or server-mode use.
        """
        if params is None:
            params = SamplingParams()
        self._ensure_stop_ids(params)
        self._raise_if_loop_unhealthy()

        req = Request(
            rid=self._next_rid,
            prompt_ids=input_ids,
            sampling_params=params,
            timeout_s=timeout_s,
        )
        with self._lock:
            self._next_rid += 1
            self.scheduler.add_request(req)
        self._work_event.set()
        return req

    def chat(
        self,
        messages: list[dict[str, str]],
        params: SamplingParams | None = None,
        **template_kwargs,
    ) -> GenerationResult:
        if params is None:
            params = SamplingParams()
        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **template_kwargs,
        )
        input_ids = self.tokenizer.encode(formatted, add_special_tokens=False)
        return self.generate(input_ids, params)

    def complete(
        self,
        prompt: str,
        params: SamplingParams | None = None,
    ) -> GenerationResult:
        if params is None:
            params = SamplingParams()
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()
        return self.generate(input_ids, params)

    def generate(
        self,
        input_ids: list[int],
        params: SamplingParams | None = None,
    ) -> GenerationResult:
        with record_function("serve.generate"):
            req = self.submit(input_ids, params)

            if self._has_server_loop():
                req._done_event.wait()
            else:
                while not req.is_finished:
                    self._step()

            return self._to_result(req)

    def generate_batch(
        self,
        prompts: list[list[int]],
        params: SamplingParams | None = None,
    ) -> list[GenerationResult]:
        """Generate for multiple prompts concurrently."""
        with record_function("serve.generate_batch"):
            if params is None:
                params = SamplingParams()
            self._ensure_stop_ids(params)

            reqs = [self.submit(ids, params) for ids in prompts]

            if self._has_server_loop():
                for r in reqs:
                    r._done_event.wait()
            else:
                while not all(r.is_finished for r in reqs):
                    self._step()

            return [self._to_result(r) for r in reqs]

    def generate_stream(
        self,
        input_ids: list[int],
        params: SamplingParams | None = None,
    ):
        req = self.submit(input_ids, params)

        prev_len = 0
        if self._has_server_loop():
            # Server loop is stepping — wait on token events.
            while not req.is_finished:
                req._token_event.wait()
                req._token_event.clear()
                for tok_id in req.output_ids[prev_len:]:
                    text = self.tokenizer.decode([tok_id], skip_special_tokens=True)
                    yield tok_id, text, self._to_result(req)
                prev_len = len(req.output_ids)
        else:
            # CLI mode — drive the step loop ourselves.
            while not req.is_finished:
                self._step()
                if len(req.output_ids) > prev_len:
                    for tok_id in req.output_ids[prev_len:]:
                        text = self.tokenizer.decode([tok_id], skip_special_tokens=True)
                        yield tok_id, text, self._to_result(req)
                    prev_len = len(req.output_ids)

    def chat_stream(self, messages, params=None, **template_kwargs):
        if params is None:
            params = SamplingParams()
        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **template_kwargs,
        )
        input_ids = self.tokenizer.encode(formatted, add_special_tokens=False)
        yield from self.generate_stream(input_ids, params)

    # -- server mode (background step loop) --------------------------------

    def start_server_loop(self) -> None:
        """Start a background thread that continuously steps the scheduler.

        Used by the API server so multiple requests can be submitted
        concurrently and the step loop processes them all.
        """
        if hasattr(self, '_loop_thread') and self._loop_thread.is_alive():
            return
        self._loop_error = None
        self._loop_running = True
        self._loop_thread = threading.Thread(target=self._server_loop, daemon=True)
        self._loop_thread.start()

    def stop_server_loop(self) -> None:
        self._loop_running = False
        if hasattr(self, '_loop_thread'):
            self._loop_thread.join(timeout=5.0)

    def _has_server_loop(self) -> bool:
        return (
            self._loop_running
            and hasattr(self, '_loop_thread')
            and self._loop_thread.is_alive()
        )

    def _server_loop(self) -> None:
        """Background loop: step while there's work, wait for signal otherwise."""
        torch.set_grad_enabled(False)
        try:
            while self._loop_running:
                self._raise_if_loop_unhealthy()
                with self._lock:
                    has_work = self.scheduler.has_work
                if has_work:
                    self._step()
                else:
                    self._work_event.wait(timeout=0.1)
                    self._work_event.clear()
        except Exception as exc:
            self._loop_error = f"{type(exc).__name__}: {exc}"
            with self._lock:
                self.scheduler.fail_all("engine_error")
        finally:
            self._loop_running = False
            self._work_event.set()

    # -- core step ---------------------------------------------------------

    def _step(self) -> None:
        """One scheduling + forward + sample step.

        In TP mode, broadcasts step info to followers before the forward
        pass so they can mirror it.
        """
        with record_function("serve.step"):
            with record_function("serve.scheduler_step"):
                with self._lock:
                    batch = self.scheduler.step()
                    ssm_control_ops = self.scheduler.drain_ssm_control_ops()
        if batch is None:
            if self.world_size > 1:
                with record_function("serve.tp_idle_broadcast"):
                    self._broadcast_ssm_control_ops(ssm_control_ops)
                    self._broadcast_step_idle()
            return

        # Broadcast batch to followers.
        if self.world_size > 1:
            with record_function("serve.tp_batch_broadcast"):
                self._broadcast_ssm_control_ops(ssm_control_ops)
                self._broadcast_step(batch)

        # Build SSM cache indices if model has SSM layers.
        ssm_indices = None
        if self.scheduler.ssm_pool is not None:
            with record_function("serve.build_ssm_indices"):
                ssm_indices = torch.tensor(
                    [r.ssm_slot for r in batch.requests],
                    dtype=torch.int64, device=self.device,
                )

        with record_function(f"serve.forward_batch.{batch.mode}"):
            logits = self.runner.forward_batch(
                batch.token_ids, batch.q_seqlens_t,
                batch.page_table, batch.cache_seqlens,
                mode=batch.mode,
                graph_bs=batch.graph_bs,
                ssm_cache_indices=ssm_indices,
            )

        step_log_path = os.environ.get("B12X_STEP_LOG_PATH")
        if step_log_path and batch.mode == "prefill":
            try:
                topk = torch.topk(logits[0].float(), k=10)
                payload = {
                    "mode": batch.mode,
                    "q_seqlens": batch.q_seqlens,
                    "topk_ids": topk.indices.tolist(),
                    "topk_tokens": [
                        self.tokenizer.decode([tok], skip_special_tokens=True)
                        for tok in topk.indices.tolist()
                    ],
                    "topk_logits": [float(x) for x in topk.values.tolist()],
                }
                with open(step_log_path, "w") as f:
                    json.dump(payload, f, indent=2)
            except Exception:
                pass

        if batch.mode == "prefill" and not batch.is_last_chunk:
            with record_function("serve.process_prefill_chunk.partial"):
                with self._lock:
                    self.scheduler.process_prefill_chunk(None, batch.requests)
            return

        with record_function("serve.collect_sampling_inputs"):
            params_list = [r.sampling_params for r in batch.requests]
            gen_ids = [list(r.output_ids) for r in batch.requests]
        with record_function("serve.sample_batch"):
            next_tokens = sample_batch(logits, params_list, gen_ids)
        with record_function("serve.sample_to_list"):
            token_list = next_tokens.tolist()

        if batch.mode == "prefill":
            with record_function("serve.process_prefill_chunk.final"):
                with self._lock:
                    self.scheduler.process_prefill_chunk(token_list, batch.requests)
        else:
            with record_function("serve.process_decode_output"):
                with self._lock:
                    self.scheduler.process_decode_output(token_list)

    # -- helpers -----------------------------------------------------------

    def _ensure_stop_ids(self, params: SamplingParams) -> None:
        ids = set(params.stop_token_ids or [])
        # Add all EOS-like tokens.
        if self.tokenizer.eos_token_id is not None:
            ids.add(self.tokenizer.eos_token_id)
        # Also check generation_config for additional EOS tokens.
        gen_cfg = getattr(self.tokenizer, '_tokenizer', None)
        if hasattr(self, '_extra_stop_ids'):
            ids.update(self._extra_stop_ids)
        elif not hasattr(self, '_extra_stop_ids'):
            extra = set()
            # Add <|endoftext|> and <|im_end|> if present.
            for name in ['<|endoftext|>', '<|im_end|>']:
                tok_id = self.tokenizer.convert_tokens_to_ids(name)
                if tok_id is not None and tok_id != self.tokenizer.unk_token_id:
                    extra.add(tok_id)
            self._extra_stop_ids = extra
            ids.update(extra)
        params.stop_token_ids = list(ids)

    def _raise_if_loop_unhealthy(self) -> None:
        if self._loop_error is not None:
            raise RuntimeError(f"server loop unhealthy: {self._loop_error}")
        tp_health = self._refresh_tp_worker_health()
        if tp_health is not None and tp_health["fatal"]:
            summary = tp_health["summary"] or "unknown follower failure"
            raise RuntimeError(f"tp workers unhealthy: {summary}")

    def server_loop_health(self) -> dict:
        tp_health = self._refresh_tp_worker_health()
        loop_alive = hasattr(self, '_loop_thread') and self._loop_thread.is_alive()
        fatal = self._loop_error is not None or (tp_health is not None and tp_health["fatal"])
        degraded = not fatal and self._loop_running and not loop_alive
        return {
            "status": "fatal" if fatal else ("degraded" if degraded else "ok"),
            "running": self._loop_running,
            "alive": loop_alive,
            "ready": not fatal and (not self._loop_running or loop_alive),
            "healthy": not fatal,
            "degraded": degraded,
            "fatal": fatal,
            "last_error": self._loop_error,
            "tp_workers": tp_health,
        }

    def runtime_policy(self) -> dict:
        return dict(self._runtime_policy)

    def _refresh_tp_worker_health(self) -> dict | None:
        if self.world_size <= 1:
            self._tp_health = None
            return None
        self._tp_health = get_active_tp_health()
        return self._tp_health

    def _to_result(self, req: Request) -> GenerationResult:
        return GenerationResult(
            request_id=req.rid,
            prompt_ids=req.prompt_ids,
            generated_ids=list(req.output_ids),
            finished=req.is_finished,
            finish_reason=req.finished_reason,
            time_to_first_token_ms=req.ttft_ms or 0.0,
            total_time_ms=req.total_time_ms or 0.0,
        )

    # -- follower loop (non-rank-0) ----------------------------------------

    def run_follower(self) -> None:
        """Mirror rank 0's forward passes. No scheduler or sampling needed."""
        assert self.rank != 0
        while True:
            self._apply_ssm_control_ops(self._receive_ssm_control_ops())
            result = self._receive_step()
            if result == "shutdown":
                break
            if result is None:
                continue  # Idle step — loop back.
            mode, token_ids, q_seqlens, page_table, cache_seqlens, graph_bs, ssm_indices = result
            self.runner.forward_batch(
                token_ids, q_seqlens, page_table, cache_seqlens,
                mode=mode, graph_bs=graph_bs,
                ssm_cache_indices=ssm_indices,
            )

    def shutdown(self) -> None:
        self.stop_server_loop()
        if self.rank == 0 and self.world_size > 1:
            self._broadcast_ssm_control_ops([])
            # Send shutdown signal.
            header = torch.tensor([255, 0, 0, 0, 0], dtype=torch.long, device=self.device)
            dist.broadcast(header, src=0)

    # -- TP step-level broadcast -------------------------------------------

    # Header: [mode_code, total_q, bs, graph_bs, page_table_width]
    # mode_code: 0=idle, 1=prefill, 2=decode, 255=shutdown.

    _MODE_IDLE = 0
    _MODE_PREFILL = 1
    _MODE_DECODE = 2
    _MODE_SHUTDOWN = 255

    def _broadcast_ssm_control_ops(self, ops: list[tuple[int, int, int, int]]) -> None:
        with record_function("serve.tp_broadcast_ssm_ops"):
            header = torch.tensor([len(ops)], dtype=torch.long, device=self.device)
            dist.broadcast(header, src=0)
            if not ops:
                return
            payload = torch.tensor(ops, dtype=torch.long, device=self.device)
            dist.broadcast(payload, src=0)

    def _receive_ssm_control_ops(self) -> list[tuple[int, int, int, int]]:
        header = torch.empty(1, dtype=torch.long, device=self.device)
        dist.broadcast(header, src=0)
        count = header[0].item()
        if count == 0:
            return []
        payload = torch.empty(count, 4, dtype=torch.long, device=self.device)
        dist.broadcast(payload, src=0)
        return [tuple(int(v) for v in row.tolist()) for row in payload]

    def _apply_ssm_control_ops(self, ops: list[tuple[int, int, int, int]]) -> None:
        if not ops or self.runner.ssm_pool is None:
            return
        arena = self.runner.ssm_pool
        for op, a, b, _c in ops:
            if op == BatchScheduler.SSM_ALLOC_LIVE:
                slot = arena.alloc(1)[0].item()
                if slot != a:
                    raise RuntimeError(f"live SSM slot mismatch on follower: expected {a}, got {slot}")
            elif op == BatchScheduler.SSM_RESTORE_SNAPSHOT:
                arena.restore_snapshot(a, b)
            elif op == BatchScheduler.SSM_CAPTURE_SNAPSHOT:
                snapshot_slot = arena.capture_snapshot(a)
                if snapshot_slot != b:
                    raise RuntimeError(
                        f"snapshot slot mismatch on follower: expected {b}, got {snapshot_slot}"
                    )
            elif op == BatchScheduler.SSM_FREE_LIVE:
                arena.free_slot(a)
            elif op == BatchScheduler.SSM_FREE_SNAPSHOT:
                arena.free_snapshot(a)
            else:
                raise RuntimeError(f"unknown SSM control op {op}")

    def _broadcast_step_idle(self) -> None:
        """Tell followers there's no work this step."""
        header = torch.tensor([self._MODE_IDLE, 0, 0, 0, 0],
                              dtype=torch.long, device=self.device)
        dist.broadcast(header, src=0)

    def _broadcast_step(self, batch) -> None:
        """Broadcast batch tensors to followers before forward pass."""
        mode_code = self._MODE_PREFILL if batch.mode == "prefill" else self._MODE_DECODE
        total_q = batch.token_ids.shape[0]
        bs = batch.cache_seqlens.shape[0]
        graph_bs = batch.graph_bs or 0
        page_table_width = batch.page_table.shape[1]

        with record_function("serve.tp_build_header"):
            header = torch.tensor([mode_code, total_q, bs, graph_bs, page_table_width],
                                  dtype=torch.long, device=self.device)
        with record_function("serve.tp_broadcast_header"):
            dist.broadcast(header, src=0)

        # Broadcast batch tensors.
        with record_function("serve.tp_broadcast_token_ids"):
            dist.broadcast(batch.token_ids, src=0)
        with record_function("serve.tp_broadcast_q_seqlens"):
            dist.broadcast(batch.q_seqlens_t, src=0)
        with record_function("serve.tp_broadcast_page_table"):
            dist.broadcast(batch.page_table, src=0)
        with record_function("serve.tp_broadcast_cache_seqlens"):
            dist.broadcast(batch.cache_seqlens, src=0)

        # Broadcast SSM cache indices if hybrid model.
        if self.scheduler.ssm_pool is not None:
            with record_function("serve.tp_build_ssm_idx"):
                ssm_idx = torch.tensor(
                    [r.ssm_slot for r in batch.requests],
                    dtype=torch.int64, device=self.device,
                )
            with record_function("serve.tp_broadcast_ssm_idx"):
                dist.broadcast(ssm_idx, src=0)

    def _receive_step(self):
        """Receive a step broadcast from rank 0.

        Returns "shutdown", None (idle), or (mode, token_ids, ...) tuple.
        """
        header = torch.empty(5, dtype=torch.long, device=self.device)
        dist.broadcast(header, src=0)

        mode_code = header[0].item()
        if mode_code == self._MODE_SHUTDOWN:
            return "shutdown"
        if mode_code == self._MODE_IDLE:
            return None

        total_q = header[1].item()
        bs = header[2].item()
        graph_bs = header[3].item() or None
        page_table_width = header[4].item()
        mode = "prefill" if mode_code == self._MODE_PREFILL else "decode"

        token_ids = torch.empty(total_q, dtype=torch.long, device=self.device)
        dist.broadcast(token_ids, src=0)
        q_seqlens_t = torch.empty(bs, dtype=torch.int32, device=self.device)
        dist.broadcast(q_seqlens_t, src=0)

        page_table = torch.empty(bs, page_table_width, dtype=torch.int32, device=self.device)
        dist.broadcast(page_table, src=0)
        cache_seqlens = torch.empty(bs, dtype=torch.int32, device=self.device)
        dist.broadcast(cache_seqlens, src=0)

        # Receive SSM indices if hybrid model.
        ssm_indices = None
        if self.runner.ssm_pool is not None:
            ssm_indices = torch.empty(bs, dtype=torch.int64, device=self.device)
            dist.broadcast(ssm_indices, src=0)

        return mode, token_ids, q_seqlens_t, page_table, cache_seqlens, graph_bs, ssm_indices
