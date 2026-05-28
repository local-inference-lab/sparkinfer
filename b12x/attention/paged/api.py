"""Public isolated API for the primary paged-attention backend."""

from __future__ import annotations

from functools import lru_cache
import os

import cuda.bindings.driver as cuda
import cutlass
import torch
from cutlass.cute.runtime import from_dlpack

from b12x.cute.compiler import KernelCompileSpec, launch as b12x_launch
from b12x.cute.utils import current_cuda_stream

from .forward_paged import (
    PagedForwardKernel,
)
from .forward_extend_generic import build_extend_forward_kernel
from .merge import PagedPersistentMergeKernel, default_paged_persistent_ctas
from .traits import PagedForwardTraits, select_paged_forward_traits_from_plan
from .workspace import PagedAttentionWorkspace

_DECODE_NATIVE_FP8_QKV_MAX_SMALL_BATCH = 2
_DECODE_NATIVE_FP8_QKV_MIN_LONG_CHUNK_PAGES = 11


def _turbo_attention_enabled() -> bool:
    return os.environ.get("B12X_TURBO_ATTN") == "1"


def _torch_to_cutlass_dtype(dtype: torch.dtype) -> type[cutlass.Numeric]:
    if dtype == torch.bfloat16:
        return cutlass.BFloat16
    if dtype == torch.float16:
        return cutlass.Float16
    if dtype == torch.float8_e4m3fn:
        return cutlass.Float8E4M3FN
    if dtype == torch.float32:
        return cutlass.Float32
    raise TypeError(f"unsupported dtype {dtype}")


def _torch_to_cutlass_storage_dtype(dtype: torch.dtype) -> type[cutlass.Numeric]:
    if dtype == torch.float8_e4m3fn:
        return cutlass.Uint8
    return _torch_to_cutlass_dtype(dtype)


def _to_kernel_tensor(
    tensor: torch.Tensor | None,
    dtype: type[cutlass.Numeric],
    *,
    assumed_align: int = 16,
) -> torch.Tensor | cutlass.cute.Tensor | None:
    if tensor is None:
        return None
    cute_tensor = from_dlpack(tensor, assumed_align=assumed_align)
    cute_tensor.element_type = dtype
    leading_dim = next(
        (idx for idx, stride in enumerate(tensor.stride()) if stride == 1), None
    )
    if leading_dim is not None and tensor.ndim >= 2:
        cute_tensor = cute_tensor.mark_layout_dynamic(leading_dim=leading_dim)
    return cute_tensor


def _as_int32_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor if tensor.dtype == torch.int32 else tensor.to(torch.int32)


def _uses_native_fp8_attention_mma(
    *,
    plan,
) -> bool:
    return _turbo_attention_enabled() and plan.kv_dtype == torch.float8_e4m3fn


def _resolve_native_fp8_attention_mma_flags(
    *,
    plan,
) -> tuple[bool, bool, bool]:
    use_native_fp8_qk = _uses_native_fp8_attention_mma(plan=plan)
    decode_runtime_chunk_guard = False
    if use_native_fp8_qk and plan.mode in ("decode", "verify"):
        if (
            plan.total_q > _DECODE_NATIVE_FP8_QKV_MAX_SMALL_BATCH
            and plan.kv_chunk_size
            < _DECODE_NATIVE_FP8_QKV_MIN_LONG_CHUNK_PAGES * plan.page_size
        ):
            # Mid-batch short-chunk decode sees the native FP8 QK loss without enough replay gain.
            use_native_fp8_qk = False
    use_native_fp8_pv = (
        use_native_fp8_qk
        and plan.mode in ("decode", "verify")
        and plan.kv_chunk_size <= 384
    )
    return use_native_fp8_qk, use_native_fp8_pv, decode_runtime_chunk_guard


@lru_cache(maxsize=16)
def _dummy_plane_tma_desc_ptrs(device_index: int, num_heads: int) -> torch.Tensor:
    return torch.zeros(
        (num_heads,), dtype=torch.int64, device=torch.device("cuda", device_index)
    )


def _encode_plane_tma_descriptors(
    cache: torch.Tensor,
    *,
    plane_cols: int,
    tile_rows: int | None = None,
) -> torch.Tensor:
    if cache.ndim != 4:
        raise ValueError(
            "cache must have shape [num_pages, page_size, kv_heads, head_dim]"
        )
    num_pages, page_size, kv_heads, head_dim = [int(dim) for dim in cache.shape]
    if plane_cols <= 0 or head_dim % plane_cols != 0:
        raise ValueError(f"plane_cols={plane_cols} must divide head_dim={head_dim}")
    if tile_rows is None:
        tile_rows = page_size
    if tile_rows <= 0 or page_size % tile_rows != 0:
        raise ValueError(
            f"tile_rows={tile_rows} must be positive and divide page_size={page_size}"
        )

    swizzle_name = os.environ.get("B12X_PAGED_KV_TMA_PLANE_SWIZZLE", "")
    swizzle = (
        cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE
        if swizzle_name == "none"
        else cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B
    )
    if cache.dtype == torch.float8_e4m3fn:
        data_type = cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8
        elem_bytes = 1
    elif cache.dtype == torch.bfloat16:
        data_type = cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16
        elem_bytes = 2
    elif cache.dtype == torch.float16:
        data_type = cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_FLOAT16
        elem_bytes = 2
    else:
        raise TypeError(f"unsupported plane TMA cache dtype {cache.dtype}")
    U64 = cuda.cuuint64_t
    U32 = cuda.cuuint32_t
    row_bytes = kv_heads * head_dim * elem_bytes
    total_rows = num_pages * page_size
    base_ptr = int(cache.view(torch.uint8).data_ptr())
    head_stride_bytes = head_dim * elem_bytes

    host_desc = torch.empty((kv_heads, 16), dtype=torch.uint64)
    for kv_head_idx in range(kv_heads):
        result, tensor_map = cuda.cuTensorMapEncodeTiled(
            data_type,
            2,
            base_ptr + kv_head_idx * head_stride_bytes,
            [U64(head_dim), U64(total_rows)],
            [U64(row_bytes)],
            [U32(plane_cols), U32(tile_rows)],
            [U32(1), U32(1)],
            cuda.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
            swizzle,
            cuda.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_NONE,
            cuda.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        )
        if result != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuTensorMapEncodeTiled failed: {result}")
        host_desc[kv_head_idx] = torch.tensor(
            [int(word) for word in tensor_map.opaque],
            dtype=torch.uint64,
        )
    return host_desc.to(device=cache.device, non_blocking=False)


def _descriptor_row_ptrs(desc: torch.Tensor) -> torch.Tensor:
    row_bytes = int(desc.stride(0)) * desc.element_size()
    base_ptr = int(desc.data_ptr())
    ptrs = [base_ptr + idx * row_bytes for idx in range(int(desc.shape[0]))]
    return torch.tensor(ptrs, dtype=torch.int64, device=desc.device)


def _get_cached_plane_tma_descs(
    workspace: PagedAttentionWorkspace,
    *,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    plane_cols: int,
    tile_rows: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor, torch.Tensor] | None:
    cached = getattr(workspace, "_live_plane_tma_desc_cache", None)
    if cached is None:
        return None
    key = (
        int(k_cache.data_ptr()),
        int(v_cache.data_ptr()),
        tuple(k_cache.shape),
        tuple(v_cache.shape),
        plane_cols,
        tile_rows,
    )
    return cached.get(key)


def _tensor_meta_key(
    tensor: torch.Tensor | None,
) -> tuple[tuple[int, ...], tuple[int, ...], str, tuple[str, int | None]] | None:
    if tensor is None:
        return None
    return (
        tuple(tensor.shape),
        tuple(tensor.stride()),
        str(tensor.dtype),
        (tensor.device.type, tensor.device.index),
    )


@lru_cache(maxsize=64)
def _build_forward_kernel(
    traits: PagedForwardTraits,
    gqa_group_size: int,
    split_kv: bool,
    single_request_decode_graph: bool,
    single_qtile_decode_graph: bool,
    regularized_decode_graph: bool,
    use_native_fp8_qk: bool,
    use_native_fp8_pv: bool,
    decode_only: bool,
    decode_native_fp8_runtime_chunk_guard: bool,
    window_left: int,
    has_attention_sink_bias: bool,
) -> PagedForwardKernel:
    return PagedForwardKernel(
        _torch_to_cutlass_dtype(traits.q_dtype),
        _torch_to_cutlass_dtype(traits.kv_dtype),
        _torch_to_cutlass_storage_dtype(traits.kv_dtype),
        _torch_to_cutlass_dtype(traits.o_dtype),
        traits=traits,
        gqa_group_size=gqa_group_size,
        split_kv=split_kv,
        single_request_decode_graph=single_request_decode_graph,
        single_qtile_decode_graph=single_qtile_decode_graph,
        regularized_decode_graph=regularized_decode_graph,
        use_native_fp8_qk=use_native_fp8_qk,
        use_native_fp8_pv=use_native_fp8_pv,
        decode_only=decode_only,
        decode_native_fp8_runtime_chunk_guard=decode_native_fp8_runtime_chunk_guard,
        window_left=window_left,
        has_attention_sink_bias=has_attention_sink_bias,
    )


@lru_cache(maxsize=32)
def _build_extend_forward_kernel(
    traits: PagedForwardTraits,
    use_native_fp8_qk: bool,
    use_native_fp8_pv: bool,
    window_left: int,
    has_attention_sink_bias: bool,
) -> object:
    return build_extend_forward_kernel(
        traits,
        use_native_fp8_qk,
        use_native_fp8_pv,
        window_left=window_left,
        has_attention_sink_bias=has_attention_sink_bias,
    )


@lru_cache(maxsize=16)
def _build_merge_kernel(
    dtype: torch.dtype,
    head_dim: int,
    total_q: int,
    persistent_ctas: int,
    direct_grid: bool,
    regular_decode_graph: bool,
    pair_bf16_partial_loads: bool,
) -> PagedPersistentMergeKernel:
    cutlass_dtype = _torch_to_cutlass_dtype(dtype)
    merge_bdy = 3 if dtype == torch.bfloat16 and head_dim == 128 and regular_decode_graph else 4
    if dtype == torch.bfloat16 and head_dim == 128 and regular_decode_graph and int(total_q) == 4:
        merge_bdy = 4
    return PagedPersistentMergeKernel(
        cutlass_dtype,
        cutlass_dtype,
        head_dim=head_dim,
        vec_size=head_dim // 32,
        bdy=merge_bdy,
        persistent_ctas=persistent_ctas,
        direct_grid=direct_grid,
        regular_decode_graph=regular_decode_graph,
        pair_bf16_partial_loads=pair_bf16_partial_loads,
    )


def _resolve_paged_attention_binding(
    *,
    binding,
    q: torch.Tensor | None,
    k_cache: torch.Tensor | None,
    v_cache: torch.Tensor | None,
    workspace: PagedAttentionWorkspace | None,
    output: torch.Tensor | None,
    k_descale: torch.Tensor | None,
    v_descale: torch.Tensor | None,
    attention_sink_bias: torch.Tensor | None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    PagedAttentionWorkspace,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    if binding is None:
        missing = [
            name
            for name, value in (
                ("q", q),
                ("k_cache", k_cache),
                ("v_cache", v_cache),
                ("workspace", workspace),
                ("output", output),
            )
            if value is None
        ]
        if missing:
            raise TypeError(f"missing required paged attention arguments: {', '.join(missing)}")
        return q, k_cache, v_cache, workspace, output, k_descale, v_descale, attention_sink_bias

    extras = [
        name
        for name, value in (
            ("q", q),
            ("k_cache", k_cache),
            ("v_cache", v_cache),
            ("workspace", workspace),
            ("output", output),
            ("k_descale", k_descale),
            ("v_descale", v_descale),
            ("attention_sink_bias", attention_sink_bias),
        )
        if value is not None
    ]
    if extras:
        raise ValueError(
            "paged attention binding owns runtime tensors and workspace; "
            f"do not also pass {', '.join(extras)}"
        )
    return (
        binding.q,
        binding.k_cache,
        binding.v_cache,
        binding.workspace,
        binding.output,
        binding.k_descale,
        binding.v_descale,
        binding.attention_sink_bias,
    )


def paged_attention_forward(
    q: torch.Tensor | None = None,
    k_cache: torch.Tensor | None = None,
    v_cache: torch.Tensor | None = None,
    *,
    workspace: PagedAttentionWorkspace | None = None,
    output: torch.Tensor | None = None,
    k_descale: torch.Tensor | None = None,
    v_descale: torch.Tensor | None = None,
    attention_sink_bias: torch.Tensor | None = None,
    binding=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    q, k_cache, v_cache, workspace, output, k_descale, v_descale, attention_sink_bias = (
        _resolve_paged_attention_binding(
            binding=binding,
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            workspace=workspace,
            output=output,
            k_descale=k_descale,
            v_descale=v_descale,
            attention_sink_bias=attention_sink_bias,
        )
    )
    plan = workspace.plan
    page_table = workspace.page_table
    cache_seqlens = workspace.cache_seqlens
    cu_seqlens_q = workspace.cu_seqlens_q
    if page_table is None or cache_seqlens is None or cu_seqlens_q is None:
        raise RuntimeError("paged workspace metadata has not been prepared")
    if plan.split_kv and (workspace.tmp_output is None or workspace.tmp_lse is None):
        raise ValueError(
            "split-kv plan requires tmp_output and tmp_lse in the workspace"
        )
    if k_descale is not None and k_descale.ndim == 2 and int(k_descale.shape[1]) == 1:
        k_descale = k_descale[:, 0].contiguous()
    if v_descale is not None and v_descale.ndim == 2 and int(v_descale.shape[1]) == 1:
        v_descale = v_descale[:, 0].contiguous()
    if output.ndim != 3:
        raise ValueError(
            f"output must be rank-3 [total_q, heads, head_dim], got {tuple(output.shape)}"
        )
    if int(output.shape[0]) < int(plan.total_q):
        raise ValueError(
            f"output first dimension must be at least total_q={plan.total_q}, got {int(output.shape[0])}"
        )
    if tuple(output.shape[1:]) != (plan.num_q_heads, plan.head_dim_vo):
        raise ValueError(
            "output shape must match the prepared workspace contract: "
            f"expected (*, {plan.num_q_heads}, {plan.head_dim_vo}), got {tuple(output.shape)}"
        )

    if (
        k_cache.dtype == torch.float8_e4m3fn or v_cache.dtype == torch.float8_e4m3fn
    ) and (k_descale is None or v_descale is None):
        raise ValueError("fp8 paged caches require k_descale and v_descale")
    if workspace.kv_window_start_tokens is None:
        raise RuntimeError("paged workspace is missing kv_window_start_tokens")
    has_attention_sink_bias = attention_sink_bias is not None
    if attention_sink_bias is None:
        attention_sink_bias = torch.empty(0, dtype=torch.float32, device=q.device)
    else:
        if attention_sink_bias.ndim != 1:
            raise ValueError(
                f"attention_sink_bias must be rank-1 [num_q_heads], got {tuple(attention_sink_bias.shape)}"
            )
        if int(attention_sink_bias.shape[0]) != plan.num_q_heads:
            raise ValueError(
                f"attention_sink_bias must have {plan.num_q_heads} elements, got {int(attention_sink_bias.shape[0])}"
            )
        if attention_sink_bias.device != q.device:
            raise ValueError("attention_sink_bias must be on the same CUDA device as q")
        if attention_sink_bias.dtype != torch.float32:
            attention_sink_bias = attention_sink_bias.to(torch.float32)
        if not attention_sink_bias.is_contiguous():
            attention_sink_bias = attention_sink_bias.contiguous()

    traits = select_paged_forward_traits_from_plan(plan)
    use_native_fp8_qk, use_native_fp8_pv, decode_native_fp8_runtime_chunk_guard = (
        _resolve_native_fp8_attention_mma_flags(plan=plan)
    )
    if plan.mode == "extend":
        if plan.split_kv:
            raise ValueError("extend plans no longer support split-kv")
        forward_kernel = _build_extend_forward_kernel(
            traits,
            use_native_fp8_qk,
            use_native_fp8_pv,
            plan.window_left,
            has_attention_sink_bias,
        )
    else:
        disable_single_request_decode_graph = os.environ.get(
            "B12X_PAGED_DISABLE_SINGLE_REQUEST_DECODE_GRAPH", "0"
        ).lower() in {"1", "true", "yes", "on"}
        single_request_decode_graph = (
            plan.mode == "decode"
            and plan.enable_cuda_graph
            and plan.split_kv
            and plan.gqa_group_size <= 8
            and workspace._decode_graph_chunk_pages_lut is not None
            and plan.num_qo_tiles == 1
            and plan.page_table_shape[0] == 1
            and not disable_single_request_decode_graph
        )
        single_qtile_decode_graph = (
            plan.mode == "decode"
            and plan.enable_cuda_graph
            and plan.split_kv
            and plan.gqa_group_size <= 8
            and workspace._decode_graph_chunk_pages_lut is not None
            and plan.page_table_shape[0] > 1
            and max(plan.qo_tile_indices, default=0) == 0
        )
        regularized_decode_graph = bool(
            single_qtile_decode_graph and workspace._use_regular_decode_graph_replay
        )
        forward_kernel = _build_forward_kernel(
            traits,
            plan.gqa_group_size,
            plan.split_kv,
            single_request_decode_graph,
            single_qtile_decode_graph,
            regularized_decode_graph,
            use_native_fp8_qk,
            use_native_fp8_pv,
            plan.mode == "decode",
            decode_native_fp8_runtime_chunk_guard,
            plan.window_left,
            has_attention_sink_bias,
        )
    forward_output = workspace.tmp_output if plan.split_kv else output
    forward_lse = workspace.tmp_lse if plan.split_kv else workspace.lse
    assert forward_output is not None
    assert forward_lse is not None

    q_arg = _to_kernel_tensor(q, _torch_to_cutlass_dtype(q.dtype))
    k_cache_arg = (
        _to_kernel_tensor(k_cache.view(torch.uint8), cutlass.Uint8)
        if k_cache.dtype == torch.float8_e4m3fn
        else _to_kernel_tensor(k_cache, _torch_to_cutlass_dtype(k_cache.dtype))
    )
    v_cache_arg = (
        _to_kernel_tensor(v_cache.view(torch.uint8), cutlass.Uint8)
        if v_cache.dtype == torch.float8_e4m3fn
        else _to_kernel_tensor(v_cache, _torch_to_cutlass_dtype(v_cache.dtype))
    )
    forward_output_arg = _to_kernel_tensor(
        forward_output, _torch_to_cutlass_dtype(forward_output.dtype)
    )
    forward_lse_arg = _to_kernel_tensor(forward_lse, cutlass.Float32)
    page_table_arg = _to_kernel_tensor(
        _as_int32_tensor(page_table), cutlass.Int32, assumed_align=4
    )
    cache_seqlens_arg = _to_kernel_tensor(
        _as_int32_tensor(cache_seqlens), cutlass.Int32, assumed_align=4
    )
    cu_seqlens_q_arg = _to_kernel_tensor(
        _as_int32_tensor(cu_seqlens_q), cutlass.Int32, assumed_align=4
    )
    request_indices_arg = _to_kernel_tensor(
        workspace.request_indices, cutlass.Int32, assumed_align=4
    )
    qo_tile_indices_arg = _to_kernel_tensor(
        workspace.qo_tile_indices, cutlass.Int32, assumed_align=4
    )
    kv_tile_indices_arg = _to_kernel_tensor(
        workspace.kv_tile_indices, cutlass.Int32, assumed_align=4
    )
    o_indptr_arg = _to_kernel_tensor(workspace.o_indptr, cutlass.Int32, assumed_align=4)
    kv_chunk_size_arg = _to_kernel_tensor(
        workspace.kv_chunk_size_ptr, cutlass.Int32, assumed_align=4
    )
    kv_window_start_arg = _to_kernel_tensor(
        workspace.kv_window_start_tokens, cutlass.Int32, assumed_align=4
    )
    block_valid_mask_arg = _to_kernel_tensor(
        workspace.block_valid_mask, cutlass.Int32, assumed_align=4
    )
    attention_sink_bias_arg = _to_kernel_tensor(
        attention_sink_bias, cutlass.Float32, assumed_align=4
    )
    k_descale_arg = _to_kernel_tensor(k_descale, cutlass.Float32)
    v_descale_arg = _to_kernel_tensor(v_descale, cutlass.Float32)
    k_tma_desc_ptrs: torch.Tensor | None = None
    v_tma_desc_ptrs: torch.Tensor | None = None
    k_tma_desc: torch.Tensor | None = None
    v_tma_desc: torch.Tensor | None = None
    if plan.mode == "extend" and (
        getattr(forward_kernel, "use_paged_kv_tma_raw_desc_issue", False)
        or getattr(forward_kernel, "use_paged_kv_tma_fp8_raw_issue", False)
    ):
        cached_descs = _get_cached_plane_tma_descs(
            workspace,
            k_cache=k_cache,
            v_cache=v_cache,
            plane_cols=forward_kernel.kv_tma_plane_head_dim,
            tile_rows=forward_kernel.stage_tile_rows,
        )
        if cached_descs is not None:
            k_tma_desc, v_tma_desc, k_tma_desc_ptrs, v_tma_desc_ptrs = cached_descs
        else:
            k_tma_desc = _encode_plane_tma_descriptors(
                k_cache,
                plane_cols=forward_kernel.kv_tma_plane_head_dim,
                tile_rows=forward_kernel.stage_tile_rows,
            )
            v_tma_desc = _encode_plane_tma_descriptors(
                v_cache,
                plane_cols=forward_kernel.kv_tma_plane_head_dim,
                tile_rows=forward_kernel.stage_tile_rows,
            )
            k_tma_desc_ptrs = _descriptor_row_ptrs(k_tma_desc)
            v_tma_desc_ptrs = _descriptor_row_ptrs(v_tma_desc)
            workspace._live_plane_tma_desc_cache[
                (
                    int(k_cache.data_ptr()),
                    int(v_cache.data_ptr()),
                    tuple(k_cache.shape),
                    tuple(v_cache.shape),
                    forward_kernel.kv_tma_plane_head_dim,
                    forward_kernel.stage_tile_rows,
                )
            ] = (
                k_tma_desc,
                v_tma_desc,
                k_tma_desc_ptrs,
                v_tma_desc_ptrs,
            )
    if k_tma_desc_ptrs is None or v_tma_desc_ptrs is None:
        dummy_desc_ptrs = _dummy_plane_tma_desc_ptrs(
            torch.cuda.current_device(),
            plan.num_kv_heads,
        )
        k_tma_desc_ptrs = dummy_desc_ptrs
        v_tma_desc_ptrs = dummy_desc_ptrs
    workspace._live_plane_tma_descs = (
        k_tma_desc,
        v_tma_desc,
        k_tma_desc_ptrs,
        v_tma_desc_ptrs,
    )

    stream = current_cuda_stream()
    use_capacity_contract = (
        plan.mode in ("extend", "verify") and workspace.fixed_capacity
    )
    q_cache_tensor = (
        workspace._plan_q
        if use_capacity_contract and workspace._plan_q is not None
        else q
    )
    output_cache_tensor = (
        workspace._plan_output
        if use_capacity_contract and workspace._plan_output is not None
        else forward_output
    )
    forward_cache_key = [
        _tensor_meta_key(q_cache_tensor),
        _tensor_meta_key(k_cache),
        _tensor_meta_key(v_cache),
        _tensor_meta_key(page_table),
        _tensor_meta_key(cache_seqlens),
        _tensor_meta_key(cu_seqlens_q),
        _tensor_meta_key(workspace.request_indices),
        _tensor_meta_key(workspace.qo_tile_indices),
        _tensor_meta_key(workspace.kv_tile_indices),
        _tensor_meta_key(workspace.o_indptr),
        _tensor_meta_key(workspace.kv_chunk_size_ptr),
        _tensor_meta_key(workspace.kv_window_start_tokens),
        _tensor_meta_key(workspace.block_valid_mask),
        _tensor_meta_key(attention_sink_bias),
        _tensor_meta_key(output_cache_tensor),
        _tensor_meta_key(forward_lse),
        _tensor_meta_key(k_descale),
        _tensor_meta_key(v_descale),
    ]
    cache_key_labels = [
        "q_contract" if use_capacity_contract else "q",
        "k_cache",
        "v_cache",
        "page_table",
        "cache_seqlens",
        "cu_seqlens_q",
        "request_indices",
        "qo_tile_indices",
        "kv_tile_indices",
        "o_indptr",
        "kv_chunk_size_ptr",
        "kv_window_start_tokens",
        "block_valid_mask",
        "attention_sink_bias",
        "output_contract" if use_capacity_contract else "forward_output",
        "forward_lse",
        "k_descale",
        "v_descale",
    ]
    forward_args = [
        q_arg,
        k_cache_arg,
        v_cache_arg,
        page_table_arg,
        cache_seqlens_arg,
        cu_seqlens_q_arg,
        request_indices_arg,
        qo_tile_indices_arg,
        kv_tile_indices_arg,
        o_indptr_arg,
        kv_chunk_size_arg,
        kv_window_start_arg,
        block_valid_mask_arg,
        attention_sink_bias_arg,
        forward_output_arg,
        forward_lse_arg,
        k_descale_arg,
        v_descale_arg,
    ]
    if plan.mode == "extend":
        k_tma_desc_arg = _to_kernel_tensor(
            k_tma_desc_ptrs, cutlass.Int64, assumed_align=8
        )
        v_tma_desc_arg = _to_kernel_tensor(
            v_tma_desc_ptrs, cutlass.Int64, assumed_align=8
        )
        forward_args.extend((k_tma_desc_arg, v_tma_desc_arg))
        forward_cache_key.extend(
            (
                _tensor_meta_key(k_tma_desc_ptrs),
                _tensor_meta_key(v_tma_desc_ptrs),
            )
        )
        cache_key_labels.extend(("k_tma_desc_ptrs", "v_tma_desc_ptrs"))
    forward_args.append(stream)
    forward_spec = KernelCompileSpec.from_key(
        "attention.paged.forward",
        1,
        tuple(forward_cache_key),
        labels=tuple(cache_key_labels),
    )
    b12x_launch(
        forward_kernel,
        compile_spec=forward_spec,
        compile_args=tuple(forward_args),
        runtime_args=tuple(forward_args),
    )

    if plan.split_kv:
        persistent_ctas = default_paged_persistent_ctas(
            total_rows=plan.total_q,
            num_heads=plan.num_q_heads,
            device=output.device,
        )
        merge_regular_decode_graph = (
            plan.mode == "decode"
            and plan.enable_cuda_graph
            and plan.gqa_group_size <= 8
            and workspace._decode_graph_chunk_pages_lut is not None
            and workspace._use_regular_decode_graph_replay
            and max(plan.qo_tile_indices, default=0) == 0
        )
        merge_direct_grid = merge_regular_decode_graph
        pair_bf16_merge_partial_loads = (
            plan.mode == "decode"
            and 2 <= int(plan.total_q) <= 4
            and output.dtype == torch.bfloat16
            and workspace.tmp_output is not None
            and workspace.tmp_output.dtype == torch.bfloat16
            and plan.head_dim_vo == 128
            and plan.gqa_group_size == 6
        )
        merge_kernel = _build_merge_kernel(
            output.dtype,
            plan.head_dim_vo,
            plan.total_q,
            persistent_ctas,
            merge_direct_grid,
            merge_regular_decode_graph,
            pair_bf16_merge_partial_loads,
        )
        tmp_output_arg = _to_kernel_tensor(
            workspace.tmp_output, _torch_to_cutlass_dtype(workspace.tmp_output.dtype)
        )
        tmp_lse_arg = _to_kernel_tensor(workspace.tmp_lse, cutlass.Float32)
        merge_indptr_arg = _to_kernel_tensor(
            workspace.merge_indptr, cutlass.Int32, assumed_align=4
        )
        merge_cache_seqlens_arg = _to_kernel_tensor(
            _as_int32_tensor(cache_seqlens), cutlass.Int32, assumed_align=4
        )
        output_arg = _to_kernel_tensor(output, _torch_to_cutlass_dtype(output.dtype))
        lse_arg = _to_kernel_tensor(workspace.lse, cutlass.Float32)
        total_num_rows_arg = (
            None
            if merge_regular_decode_graph
            else _to_kernel_tensor(
                workspace.total_num_rows_ptr, cutlass.Int32, assumed_align=4
            )
        )
        merge_args = (
            tmp_output_arg,
            tmp_lse_arg,
            merge_indptr_arg,
            merge_cache_seqlens_arg,
            kv_chunk_size_arg,
            output_arg,
            lse_arg,
            total_num_rows_arg,
        )
        merge_cache_key = (
            _tensor_meta_key(workspace.tmp_output),
            _tensor_meta_key(workspace.tmp_lse),
            _tensor_meta_key(workspace.merge_indptr),
            _tensor_meta_key(cache_seqlens),
            _tensor_meta_key(workspace.kv_chunk_size_ptr),
            _tensor_meta_key(output),
            _tensor_meta_key(workspace.lse),
            None
            if merge_regular_decode_graph
            else _tensor_meta_key(workspace.total_num_rows_ptr),
            persistent_ctas,
            merge_direct_grid,
            merge_regular_decode_graph,
            pair_bf16_merge_partial_loads,
        )
        merge_spec = KernelCompileSpec.from_key(
            "attention.paged.merge",
            1,
            merge_cache_key,
            labels=(
                "tmp_output",
                "tmp_lse",
                "merge_indptr",
                "cache_seqlens",
                "kv_chunk_size_ptr",
                "output",
                "lse",
                "total_num_rows_ptr",
                "persistent_ctas",
                "direct_grid",
                "regular_decode_graph",
                "pair_bf16_partial_loads",
            ),
        )
        b12x_launch(
            merge_kernel,
            compile_spec=merge_spec,
            compile_args=(*merge_args, stream),
            runtime_args=(*merge_args, stream),
        )

    return output[: plan.total_q], workspace.current_lse_view()


def clear_paged_caches() -> None:
    """Clear compiled-kernel caches for the primary paged backend."""
    _build_forward_kernel.cache_clear()
    _build_extend_forward_kernel.cache_clear()
    _build_merge_kernel.cache_clear()
