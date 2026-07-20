# Eager Plan → Bind → Kernel (and why bindings never own a workspace)

## The rule

> **Arenas and workspaces are sglang-only. In the vLLM path a binding must never use,
> own, or construct a workspace/arena.**

The vLLM integration is **eager**: `plan → bind → kernel`, all of which run inline
right before the kernel and are CUDA-graph-capturable. A *plan* shall never create
a workspace; a *binding* shall never create a workspace.

## Two worlds

| | sglang | vLLM |
|---|---|---|
| Memory owner | the **workspace/arena** (preplanned, long-lived, cached) | the **caller** (vLLM's `current_workspace_manager()` scratch) |
| Direction | `workspace.bind_*() → binding` (workspace makes the binding) | `plan.bind(scratch=…) → binding` (caller hands scratch in) |
| Lifetime | cached / reused across calls | constructed **fresh every call**, never cached |
| Init | workspace may pre-init/poison once | none at bind time |

These two models conflict over who owns memory, so they are kept strictly separate.
vLLM never touches the sglang side.

## Why this is always safe — eager bindings are *strictly more permissive*

A binding is built eagerly, per call, and never cached. So it can supply
**everything** a cached workspace could — and more, because it carries no
caching/lifetime/graph-stability constraints. The kernels already read their
scratch as a *bag of attributes* (`tmp_output`, `tmp_lse`, `output_buffer`,
`final_lse`, `num_chunks_ptr`, `set_split_chunk_config`, …) — they duck-type it.
A views container that exposes those same attributes is a **drop-in with zero
kernel-signature change**. Porting a workspace path to a binding therefore has, by
definition, no porting difficulty: the target API is a superset.

## What `bind()` must do (and must not do)

`bind()` maps the single caller-owned scratch tensor into the per-spec
kernel-argument **views** and returns a plain views **container**, then builds the
binding from it. Concretely it must:

- **only** `narrow()` + `view()`/`as_strided()` at precomputed byte offsets
  (`_materialize_arena_view` / `_materialize_arena_strided_view`);
- allocate **nothing** (no `torch.empty/zeros/arange` returning a new tensor — any
  allocation during CUDA-graph capture is illegal and crashes);
- **not** init-write the scratch (counters are zeroed by the kernel prologue;
  ramps/maps are write-first; the only permitted write is a *guarded in-place*
  `fill_` on a scalar control view, e.g. `set_split_chunk_config`);
- **never** construct a `SPARKINFERAttentionWorkspace` / `_TPCoreArena` / call
  `from_shared_arena` / `_make_workspace_views` / `make_workspace` /
  `_materialize_core_arena`.

Required per-call state lives where it belongs: **kernel prologue** (counter/queue
zeroing, derived ramps), **launch wrapper** (in-place `zero_()` of read-before-write
barriers), or **caller** — never a bind-time arena materialization.

## The canonical pattern

`SPARKINFERCompressedMLAScratch` (`sparkinfer/integration/compressed_scratch.py`) is the gold
template, and `SPARKINFERSparseMLAScratch` (`sparse_mla_scratch.py`) mirrors it:

```
@dataclass(kw_only=True)
class <X>Scratch:                       # a VIEWS CONTAINER, never a workspace
    shared_scratch: torch.Tensor        # the one caller-owned storage
    device/dtype/.../max_chunks_per_row # scalar caps copied off the plan
    tmp_output/tmp_lse/output_buffer/…  # views (narrow+view), nullable
    def set_split_chunk_config(...):    # guarded in-place fill_ only

def _materialize_<x>_scratch(caps, storage, layout) -> <X>Scratch:  # pure views
def <X>ScratchPlan.bind(scratch, ...) -> <X>Binding:                # no workspace
    return build_<x>_binding(scratch=_materialize_<x>_scratch(...), ...)

@dataclass(frozen=True, kw_only=True)
class <X>Binding:
    scratch: object                     # the views container — NOT a workspace
    ...                                 # plus the caller tensors
```

The caller (vLLM) allocates one scratch tensor of `plan.layout.nbytes` from its
workspace manager — co-allocated with any other live buffer (e.g. the q-concat
buffer) in a **single** `get_simultaneous` call so they get distinct,
non-overlapping offsets — and passes it to `plan.bind(scratch=…)`.

## Porting checklist (workspace path → eager binding)

1. Add an `<X>Scratch` views container exposing exactly the attributes the kernel
   reads off the (former) workspace.
2. Add `_<x>_scratch_layout` (a pure byte-cursor) + `_materialize_<x>_scratch`
   (pure `narrow`+`view`/`as_strided`, no alloc, no init).
3. Rewrite `bind()` to `scratch_tensor → _materialize_<x>_scratch → build_<x>_binding`.
   Delete `from_shared_arena` / `_make_workspace_views`.
4. Change the binding's `scratch` field type to the container (`object`).
5. Relax kernel-entry type hints from `SPARKINFERAttentionWorkspace` to `object`; the
   kernels are unchanged (they duck-type).
6. In vLLM, build the plan once, allocate one caller scratch tensor, and
   `plan.bind(...)` per forward. Never construct a `SPARKINFERAttentionWorkspace`.
