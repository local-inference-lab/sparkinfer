"""Monkey-patch sglang allreduce dispatch to use the b12x PCIe oneshot kernel.

Activated by setting ``B12X_USE_PCIE_AR=1`` in the environment.  When activated,
``apply_dsv4_allreduce_patch()`` replaces

* ``sglang.srt.distributed.communication_op.tensor_model_parallel_all_reduce``
* ``sglang.srt.distributed.parallel_state.GroupCoordinator.all_reduce``
* ``sglang.srt.distributed.parallel_state.GroupCoordinator.graph_capture``

so that BF16 allreduces on (T, 4096) with T <= 32 are handled by
:class:`b12x.distributed.PCIeOneshotAllReduce`.  All other shapes/dtypes fall
through to the original NCCL/sglang path.

v2 changes (track-A v2):
- The graph_capture wrapper now PRE-INITIALISES the b12x runtime and PRE-WARMS
  the kernel for every (shape, dtype) we expect inside graph capture.  This
  drains all IPC handle exchanges, JIT compilation and ``cudaMalloc``s out of
  the capture window, so capture never touches NCCL or allocator paths that
  invalidate ``cudaStreamCaptureStatus``.
- A persistent output-buffer pool is allocated up-front (one BF16 tensor of
  ``max_bs * 4096`` per shape) and reused on the patched call.  If a request
  arrives during capture for a shape we did NOT pre-warm, we fall through to
  the original NCCL path instead of allocating a new buffer.
- ``B12X_PCIE_AR_PREWARM_SHAPES`` env var
  (default ``"1,4096;4,4096;5,4096;8,4096;16,4096;32,4096"``)
  controls the set of shapes warmed before capture.
- ``B12X_USE_PCIE_AR_NOGRAPH=1`` remains as a final fallback that disables
  b12x while a stream is being captured.

The patch is idempotent (safe to call repeatedly) and tolerant of sglang or
torch.distributed not being initialized yet — the b12x runtime is constructed
lazily on the first eligible call.
"""

from __future__ import annotations

import logging
import os
import threading
from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch.distributed import ProcessGroup

from b12x.distributed.pcie_oneshot import (
    PCIeOneshotAllReduce,
    SUPPORTED_WORLD_SIZES,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration knobs
# ---------------------------------------------------------------------------

# Hidden dimensions for DSv4-Flash decode allreduces.  Prologue/MQA path uses
# 4096; MoE intermediate paths surface 2048 and 7168.
_DSV4_HIDDEN = 4096
_DSV4_HIDDENS_FAST_PATH = (4096,)
# Maximum number of tokens (rows) we accept on the fast path.  The standalone
# PCIe oneshot path has a verified small-T win, but the known T=64 broadening
# regressed against NCCL.  Keep this ceiling conservative unless a benchmark
# proves a new fused/overlapped path.
_DSV4_MAX_TOKENS = 32
# Default eager-buffer size — needs to fit the largest fast-path AR.
# 64 tokens * 4096 hidden * 2 bytes BF16 = 512 KB.  Round up to 1 MB.
_DEFAULT_EAGER_BYTES = 1 * 1024 * 1024  # 1 MB

# Module-level latch / singletons so that the patch is idempotent and the b12x
# runtime survives across all workers in a single process.
_PATCH_LOCK = threading.Lock()
_PATCH_APPLIED = False
_RUNTIME_LOCK = threading.Lock()
_RUNTIMES: dict[Tuple[int, int, int], "_RuntimeBundle"] = {}

_ENV_ENABLE = "B12X_USE_PCIE_AR"
_ENV_ENABLE_NOGRAPH = "B12X_USE_PCIE_AR_NOGRAPH"
_ENV_VERBOSE = "B12X_PCIE_AR_VERBOSE"
_ENV_PREWARM = "B12X_PCIE_AR_PREWARM_SHAPES"
_ENV_MAX_TOKENS = "B12X_PCIE_AR_MAX_TOKENS"
_DEFAULT_PREWARM = "1,4096;4,4096;5,4096;8,4096;16,4096;32,4096"


def _verbose() -> bool:
    return os.environ.get(_ENV_VERBOSE, "0") in ("1", "true", "yes")


def _enabled() -> bool:
    return os.environ.get(_ENV_ENABLE, "0") in ("1", "true", "yes")


def _max_tokens() -> int:
    raw = os.environ.get(_ENV_MAX_TOKENS)
    if raw is None:
        return _DSV4_MAX_TOKENS
    try:
        value = int(raw)
    except ValueError:
        return _DSV4_MAX_TOKENS
    eager_limit = _DEFAULT_EAGER_BYTES // (_DSV4_HIDDEN * 2)
    # This env var may narrow the fast path for diagnostics, but it must not
    # raise the standalone oneshot path into the known-regressing T=64 bucket.
    return max(1, min(value, _DSV4_MAX_TOKENS, eager_limit))


def _graph_disabled() -> bool:
    """If set, b12x AR is only used in eager mode (no graph capture)."""
    return os.environ.get(_ENV_ENABLE_NOGRAPH, "0") in ("1", "true", "yes")


def _parse_prewarm_shapes(value: Optional[str]) -> List[Tuple[int, int]]:
    """Parse ``"T1,H1;T2,H2;..."`` into a list of (T, H) shapes.

    Empty / malformed entries are silently skipped.  Always returns a list
    deduplicated in input order.
    """
    text = (value if value is not None else _DEFAULT_PREWARM).strip()
    if not text:
        return []
    seen: set[Tuple[int, int]] = set()
    out: List[Tuple[int, int]] = []
    for chunk in text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.split(",")
        if len(parts) != 2:
            continue
        try:
            t = int(parts[0])
            h = int(parts[1])
        except ValueError:
            continue
        if t <= 0 or h <= 0:
            continue
        key = (t, h)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _is_fast_shape(shape: Tuple[int, ...], element_size: int) -> bool:
    if len(shape) < 1:
        return False
    last = shape[-1]
    if last not in _DSV4_HIDDENS_FAST_PATH:
        return False
    rows = 1
    for s in shape[:-1]:
        rows *= s
    if rows == 0 or rows > _max_tokens():
        return False
    if rows * last * element_size > _DEFAULT_EAGER_BYTES:
        return False
    return True


def _capture_prewarm_shapes(value: Optional[str]) -> List[Tuple[int, int]]:
    """Return graph-capture warmup shapes that are eligible for oneshot AR.

    Warmup happens before capture, but warming unsupported rows is still
    misleading and can leave output buffers around for shapes the production
    predicate must not use.  In particular, never warm T=64 for the standalone
    oneshot path.
    """
    shapes = _parse_prewarm_shapes(value)
    if (1, _DSV4_HIDDEN) not in shapes:
        shapes = [(1, _DSV4_HIDDEN), *shapes]
    element_size = torch.tensor([], dtype=torch.bfloat16).element_size()
    return [shape for shape in shapes if _is_fast_shape(shape, element_size)]


# ---------------------------------------------------------------------------
# Runtime bundle: caches a PCIeOneshotAllReduce per (group, world_size, device)
# plus a per-shape persistent output buffer pool.
# ---------------------------------------------------------------------------


class _RuntimeBundle:
    """Owns a :class:`PCIeOneshotAllReduce` plus persistent output and input
    buffer pools keyed by ``(shape, dtype)``.

    Buffers are pre-allocated by :meth:`warmup` BEFORE graph capture begins so
    that the patched ``all_reduce`` call performs zero allocations even under
    ``cudaStreamCaptureStatusActive``.
    """

    def __init__(self, runtime: PCIeOneshotAllReduce):
        self.runtime = runtime
        # Output-buffer pool keyed by ``(shape, dtype)``.  Allocated up front
        # by :meth:`warmup` so subsequent allreduce calls can reuse a stable
        # device pointer (critical for cuda graph replay).
        self._out_pool: Dict[Tuple[Tuple[int, ...], torch.dtype], torch.Tensor] = {}
        # Persistent input scratch pool used during warmup so the kernel JITs
        # and registers IPC handles against device pointers that survive past
        # the warmup function.  Keeping these alive also guarantees that the
        # eager dbuf machinery has live peer pointers for the duration of the
        # patch.
        self._warmup_inputs: Dict[Tuple[Tuple[int, ...], torch.dtype], torch.Tensor] = {}
        self._warmup_done: set[Tuple[Tuple[int, ...], torch.dtype]] = set()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Buffer-pool helpers
    # ------------------------------------------------------------------

    def get_output(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        *,
        allow_alloc: bool,
    ) -> Optional[torch.Tensor]:
        """Return a cached output buffer for ``(shape, dtype)``.

        If no buffer exists and ``allow_alloc`` is False (we're inside graph
        capture) returns ``None`` so the caller falls through to NCCL.  If
        ``allow_alloc`` is True we lazily allocate one and add it to the pool.
        """
        key = (tuple(shape), dtype)
        with self._lock:
            cached = self._out_pool.get(key)
            if cached is not None:
                return cached
            if not allow_alloc:
                return None
            out = torch.empty(shape, dtype=dtype, device=self.runtime.device)
            self._out_pool[key] = out
            return out

    def has_output(self, shape: Tuple[int, ...], dtype: torch.dtype) -> bool:
        return (tuple(shape), dtype) in self._out_pool

    # ------------------------------------------------------------------
    # Warmup: drain IPC + JIT + per-shape pool allocations BEFORE capture
    # ------------------------------------------------------------------

    def warmup(self, shapes: Iterable[Tuple[int, int]], dtype: torch.dtype = torch.bfloat16) -> None:
        """Pre-init IPC handles and pre-allocate output buffers for every
        shape we may see during graph capture.

        For each shape we:
            1. Allocate a persistent input tensor (kept alive in
               ``self._warmup_inputs``).
            2. Allocate an output tensor in the pool.
            3. Run ``runtime.all_reduce`` once outside graph capture to JIT
               the kernel and ensure dbuf peer-pointer machinery is wired up.
            4. Synchronise the device so any deferred init in the kernel /
               extension is fully drained before capture starts.

        This is idempotent — re-running with the same shapes is a no-op.
        """
        device = self.runtime.device
        for t, h in shapes:
            key = ((t, h), dtype)
            with self._lock:
                if key in self._warmup_done:
                    continue
            shape = (t, h)
            # 1. Persistent input tensor (kept alive on the bundle).
            inp = torch.zeros(shape, dtype=dtype, device=device)
            # 2. Persistent output tensor in the pool.
            out = torch.empty(shape, dtype=dtype, device=device)
            with self._lock:
                self._warmup_inputs[key] = inp
                self._out_pool[key] = out
            # 3. Drive the kernel once (eager, NOT under capture).  This
            #    triggers extension JIT, dbuf slot init, kernel dispatch warm
            #    on the current stream.  We deliberately ignore numerical
            #    correctness — we only need the side-effects.
            try:
                if self.runtime.should_allreduce(inp):
                    self.runtime.all_reduce(inp, out=out)
            except Exception as exc:  # noqa: BLE001
                # If warmup itself fails we still register the shape as
                # "tried" so we don't loop, but we surface a warning.  The
                # patch will fall through to NCCL for this shape.
                logger.warning(
                    "[b12x AR] warmup failed for shape=%s dtype=%s: %s",
                    shape,
                    dtype,
                    exc,
                )
            with self._lock:
                self._warmup_done.add(key)
        # 4. Final sync: ensure every async init is committed before the
        #    caller starts a graph capture on this device.
        torch.cuda.synchronize(device=device)


def _runtime_key(group: ProcessGroup, device: torch.device) -> Tuple[int, int, int]:
    """Stable key — id() of the group object plus the device index."""
    import torch.distributed as dist

    world_size = dist.get_world_size(group=group)
    device_index = device.index if device.index is not None else 0
    return (id(group), world_size, device_index)


def _get_or_create_runtime(
    group: ProcessGroup, device: torch.device, *, allow_create: bool = True
) -> Optional[_RuntimeBundle]:
    import torch.distributed as dist

    world_size = dist.get_world_size(group=group)
    if world_size not in SUPPORTED_WORLD_SIZES:
        if _verbose():
            logger.warning(
                "[b12x AR] world size %d not supported by PCIe oneshot; falling through",
                world_size,
            )
        return None

    key = _runtime_key(group, device)
    bundle = _RUNTIMES.get(key)
    if bundle is not None:
        return bundle
    if not allow_create:
        return None

    # Construction MUST NOT run under cuda graph capture: it does IPC handle
    # exchange (NCCL CPU broadcast) and ``cudaMalloc``s.  Refuse to construct
    # if the current stream is capturing — caller should pre-warm via the
    # graph_capture wrapper instead.
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        if _verbose():
            logger.warning(
                "[b12x AR] refusing to construct runtime under stream capture; "
                "falling through to NCCL"
            )
        return None

    with _RUNTIME_LOCK:
        bundle = _RUNTIMES.get(key)
        if bundle is not None:
            return bundle
        try:
            runtime = PCIeOneshotAllReduce.from_exchange_group(
                exchange_group=group,
                device=device,
                eager_buffer_bytes=_DEFAULT_EAGER_BYTES,
                max_size=_DEFAULT_EAGER_BYTES,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[b12x AR] failed to initialise PCIe oneshot runtime (world_size=%d, device=%s): %s",
                world_size,
                device,
                exc,
            )
            return None
        bundle = _RuntimeBundle(runtime)
        _RUNTIMES[key] = bundle
        if _verbose():
            logger.info(
                "[b12x AR] PCIe oneshot runtime ready (world_size=%d, device=%s)",
                world_size,
                device,
            )
        return bundle


# ---------------------------------------------------------------------------
# Fast-path predicate
# ---------------------------------------------------------------------------


def _is_fast_path(inp: torch.Tensor) -> bool:
    if inp.dtype != torch.bfloat16:
        return False
    if inp.device.type != "cuda":
        return False
    if not inp.is_contiguous():
        return False
    return _is_fast_shape(tuple(inp.shape), inp.element_size())


# ---------------------------------------------------------------------------
# Allreduce dispatch helper
# ---------------------------------------------------------------------------


def _b12x_all_reduce(group_coordinator: Any, input_: torch.Tensor) -> Optional[torch.Tensor]:
    """Run b12x oneshot AR if eligible; return ``None`` on fall-through."""

    if group_coordinator.world_size == 1:
        return input_
    if not _is_fast_path(input_):
        return None

    capturing = torch.cuda.is_current_stream_capturing() if torch.cuda.is_available() else False
    if _graph_disabled() and capturing:
        return None

    # IPC handle exchange uses ``broadcast_object_list(..., device="cpu")`` so
    # we MUST pass the Gloo CPU group; the NCCL device_group has no CPU backend
    # and triggers ``No backend type associated with device type cpu``.
    exchange_group = getattr(group_coordinator, "cpu_group", None)
    if exchange_group is None:
        exchange_group = getattr(group_coordinator, "device_group", None)
    if exchange_group is None:
        return None
    device = getattr(group_coordinator, "device", None)
    if device is None or not isinstance(device, torch.device) or device.type != "cuda":
        return None

    # Construction is NOT permitted while a stream is being captured.  If we
    # haven't seen this group/device before AND we're inside capture, fall
    # through — the prewarm path should have been called earlier.
    bundle = _get_or_create_runtime(exchange_group, device, allow_create=not capturing)
    if bundle is None:
        return None

    runtime = bundle.runtime
    if not runtime.should_allreduce(input_):
        return None

    # Output buffer: must come from the pre-allocated pool when capturing.
    out = bundle.get_output(tuple(input_.shape), input_.dtype, allow_alloc=not capturing)
    if out is None:
        # Capture-time fall-through: shape was not pre-warmed.  Logging this
        # at WARN once per (capture, shape) would be ideal but we keep it
        # cheap and let the warning fire on every uncaptured warmup miss.
        if _verbose():
            logger.warning(
                "[b12x AR] output buffer for shape=%s dtype=%s missing during "
                "capture; falling through to NCCL",
                tuple(input_.shape),
                input_.dtype,
            )
        return None

    try:
        runtime.all_reduce(input_, out=out)
    except Exception as exc:  # noqa: BLE001
        # Surface the first failure loudly but continue via fall-through.
        logger.warning(
            "[b12x AR] PCIe oneshot allreduce failed (shape=%s, dtype=%s): %s — falling back to NCCL",
            tuple(input_.shape),
            input_.dtype,
            exc,
        )
        return None
    return out


# ---------------------------------------------------------------------------
# Graph-capture wrapper: register IPC handles after each capture
# ---------------------------------------------------------------------------


def _make_graph_capture_wrapper(original_graph_capture):
    """Wrap ``GroupCoordinator.graph_capture`` so we

    1. construct the b12x runtime up-front (out of capture),
    2. pre-warm it for every shape we expect inside capture, and
    3. register graph buffers post-capture so replays can resolve peer ptrs.
    """

    @contextmanager
    def graph_capture_wrapper(self, *args, **kwargs):
        # PCIe oneshot's IPC-handle exchange needs a Gloo CPU group;
        # ``self.device_group`` is NCCL-only.
        exchange_group = getattr(self, "cpu_group", None) or getattr(self, "device_group", None)
        device = getattr(self, "device", None)
        bundle: Optional[_RuntimeBundle] = None

        if (
            exchange_group is not None
            and isinstance(device, torch.device)
            and device.type == "cuda"
            and not _graph_disabled()
        ):
            try:
                bundle = _get_or_create_runtime(exchange_group, device, allow_create=True)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[b12x AR] runtime construction inside graph_capture wrapper failed: %s",
                    exc,
                )
                bundle = None

            if bundle is not None:
                shapes = _capture_prewarm_shapes(os.environ.get(_ENV_PREWARM))
                try:
                    bundle.warmup(shapes, dtype=torch.bfloat16)
                    if _verbose():
                        logger.info(
                            "[b12x AR] pre-warmed %d shape(s) before graph capture: %s",
                            len(shapes),
                            shapes,
                        )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "[b12x AR] warmup before graph capture failed: %s — capture "
                        "will fall through to NCCL for affected shapes",
                        exc,
                    )

        with original_graph_capture(self, *args, **kwargs) as ctx:
            try:
                yield ctx
            finally:
                # After capture, register graph buffers so future replays can
                # resolve peer pointers.  This is a CPU-side IPC handle
                # exchange (broadcast_object_list) — safe outside capture.
                if bundle is not None:
                    try:
                        bundle.runtime.register_graph_buffers()
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "[b12x AR] register_graph_buffers failed: %s",
                            exc,
                        )

    return graph_capture_wrapper


# ---------------------------------------------------------------------------
# Top-level patch entry point
# ---------------------------------------------------------------------------


def apply_dsv4_allreduce_patch(force: bool = False) -> bool:
    """Install the monkey-patch.  Returns True if the patch was applied (or
    was already applied).

    v2 also installs a graph_capture wrapper that pre-warms IPC handles and
    output buffers for every shape in ``B12X_PCIE_AR_PREWARM_SHAPES`` (default
    ``"1,4096;4,4096;5,4096"``) before sglang starts capturing.  Without this
    pre-warm the kernel's first invocation under capture allocates IPC peers
    on the fly and triggers ``cudaErrorStreamCaptureInvalidated``.
    """

    global _PATCH_APPLIED
    if not force and not _enabled():
        if _verbose():
            logger.info(
                "[b12x AR] %s not set; skipping patch install",
                _ENV_ENABLE,
            )
        return False

    with _PATCH_LOCK:
        if _PATCH_APPLIED:
            return True

        try:
            from sglang.srt.distributed import communication_op as comm_op
            from sglang.srt.distributed import parallel_state as ps
        except Exception as exc:  # noqa: BLE001
            logger.warning("[b12x AR] sglang not importable: %s", exc)
            return False

        original_tmp_ar = comm_op.tensor_model_parallel_all_reduce
        original_group_all_reduce = ps.GroupCoordinator.all_reduce
        original_graph_capture = ps.GroupCoordinator.graph_capture

        def patched_tmp_ar(input_: torch.Tensor) -> torch.Tensor:
            group = ps.get_tp_group()
            result = _b12x_all_reduce(group, input_)
            if result is not None:
                return result
            return original_tmp_ar(input_)

        def patched_group_all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
            result = _b12x_all_reduce(self, input_)
            if result is not None:
                return result
            return original_group_all_reduce(self, input_)

        comm_op.tensor_model_parallel_all_reduce = patched_tmp_ar
        ps.GroupCoordinator.all_reduce = patched_group_all_reduce
        ps.GroupCoordinator.graph_capture = _make_graph_capture_wrapper(original_graph_capture)

        # Stash originals on the modules so unpatch (if needed) can restore.
        comm_op._b12x_original_tmp_ar = original_tmp_ar
        ps.GroupCoordinator._b12x_original_all_reduce = original_group_all_reduce
        ps.GroupCoordinator._b12x_original_graph_capture = original_graph_capture

        _PATCH_APPLIED = True
        if _verbose():
            logger.info("[b12x AR] PCIe oneshot allreduce patch installed")
        return True


def is_patch_applied() -> bool:
    return _PATCH_APPLIED


def reset_runtime_cache() -> None:
    """Drop cached runtimes — primarily for tests."""
    with _RUNTIME_LOCK:
        for bundle in list(_RUNTIMES.values()):
            try:
                bundle.runtime.close()
            except Exception:  # noqa: BLE001
                pass
        _RUNTIMES.clear()


__all__ = [
    "apply_dsv4_allreduce_patch",
    "is_patch_applied",
    "reset_runtime_cache",
]
