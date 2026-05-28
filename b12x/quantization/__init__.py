"""BF16 → NVFP4 TMA quantization kernel API."""
from dataclasses import dataclass
from typing import Dict, Tuple

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.typing import AddressSpace

from b12x.cute.compiler import compile as b12x_compile
from b12x.cute.fp4 import align_up
from b12x.cute.utils import current_cuda_stream, get_max_active_clusters, get_num_sm, make_ptr
from b12x.quantization.bf16_to_fp4_tma import TestKernel, make_ptr as _standalone_make_ptr
from b12x.runtime_control import raise_if_kernel_resolution_frozen

_TILE_M = 128
_TILE_K = 128
_SF_VEC_SIZE = 16
_KERNEL_CACHE: Dict[Tuple, object] = {}


@dataclass
class BF16ToFP4TMAOutputs:
    packed_a_storage: torch.Tensor
    scale_storage: torch.Tensor
    packed_a_view: object
    sfa_ptr: object

    @property
    def packed_a_flat(self) -> torch.Tensor:
        return self.packed_a_storage.view(-1)

    @property
    def scale_flat(self) -> torch.Tensor:
        return self.scale_storage.view(-1)


def allocate_bf16_to_fp4_tma_outputs(
    M: int, K: int, *, device: torch.device = torch.device("cuda"),
) -> BF16ToFP4TMAOutputs:
    rows_pad = align_up(M, _TILE_M)
    cols_pad_sf = align_up(K // _SF_VEC_SIZE, 4)
    packed_a_storage = torch.zeros(1, M, K // 2, dtype=torch.uint8, device=device)
    scale_storage = torch.zeros(rows_pad * cols_pad_sf, dtype=torch.uint8, device=device)
    packed_a_view = packed_a_storage.permute(1, 2, 0).view(torch.float4_e2m1fn_x2)
    sfa_ptr = make_ptr(
        cutlass.Float8E4M3FN, scale_storage.data_ptr(),
        cute.AddressSpace.gmem, assumed_align=16,
    )
    return BF16ToFP4TMAOutputs(
        packed_a_storage=packed_a_storage,
        scale_storage=scale_storage,
        packed_a_view=packed_a_view,
        sfa_ptr=sfa_ptr,
    )


def compile_bf16_to_fp4_tma(M: int, K: int):
    """Compile the BF16→FP4 TMA kernel for (M, K). Returns a launch callable.

    The callable signature is: ``launch(bf16_input, global_scale, packed_a_flat, scale_flat)``
    where packed_a_flat and scale_flat come from ``BF16ToFP4TMAOutputs``.
    """
    assert M % _TILE_M == 0 and K % _TILE_K == 0
    cache_key = (M, K)
    cached = _KERNEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    ab = cutlass.Float4E2M1FN
    sf = cutlass.Float8E4M3FN
    bf = cutlass.BFloat16
    bf16_fake = cute.runtime.make_fake_compact_tensor(bf, (M, K), stride_order=(1, 0), assumed_align=16)
    gs_fake = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (1,), assumed_align=4)
    pa_fake = cute.runtime.make_fake_compact_tensor(ab, (M, K, 1), stride_order=(1, 0, 2), assumed_align=16)
    sfa_fake = _standalone_make_ptr(sf, 16, AddressSpace.gmem, assumed_align=16)
    mac = min(get_max_active_clusters(1), get_num_sm(torch.device("cuda")))
    kernel = TestKernel()
    raise_if_kernel_resolution_frozen("cute.compile", target=kernel, cache_key=cache_key)
    raw = b12x_compile(kernel, bf16_fake, gs_fake, pa_fake, sfa_fake, mac, current_cuda_stream())

    def launch(bf16_input, global_scale, packed_a_flat, scale_flat):
        pa_view = packed_a_flat.view(1, M, K // 2).permute(1, 2, 0).view(torch.float4_e2m1fn_x2)
        sfa_p = _standalone_make_ptr(sf, scale_flat.data_ptr(), AddressSpace.gmem, assumed_align=16)
        raw(bf16_input, global_scale, pa_view, sfa_p, current_cuda_stream())

    _KERNEL_CACHE[cache_key] = launch
    return launch
