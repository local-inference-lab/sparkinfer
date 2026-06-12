"""Oracle gates for the v1 W4A8 grouped MoE GEMM (W4A8GemmKernel).

Drives b12x/moe/fused/w4a8/gemm.py directly with synthetic MXFP4 experts and
MXFP8 activations (mxfp8_quantize_rows) and gates against the f32 torch
oracle ``A_deq @ B_deq.T`` per m-block expert:

- dyadic-data variants (payloads from the fragment-probe grids, power-of-two
  block scales): the f32 accumulation is exact, so the kernel's bf16 output
  must EQUAL the once-rounded oracle bit-for-bit (atol=0);
- a random-data variant gated on cosine similarity > 0.9999 against the
  quantization-true oracle (weights dequantized per
  _dequant_w4a8_weight_e8m0_k32, activations per payload * 2^(sf-127));
- valid-row masking: active_m below the padded row count must leave the
  masked C rows untouched.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import pytest
import torch
import torch.nn.functional as F

from b12x.moe.fused.reference import _dequant_w4a8_weight_e8m0_k32, _make_fp4_lut
from b12x.moe.fused.w4a8.gemm import (
    W4A8GemmKernel,
    repack_w4a8_weights,
    to_cute_u32,
)
from b12x.moe.fused.w4a8.quant import mxfp8_quantize_rows
from tests.test_w4a8_dynamic_kernel import _quantize_weight_mxfp4
from tests.test_w4a8_fragment_probe import _pack_b

from .helpers import require_sm120

_BLOCK_M = 16
_COMPILE_CACHE: dict[tuple, object] = {}


def _to_cute_i32(x: torch.Tensor) -> cute.Tensor:
    from cutlass.cute.runtime import from_dlpack

    tensor = from_dlpack(x, assumed_align=4)
    tensor.element_type = cutlass.Int32
    return tensor


def _run_gemm(
    a_q: torch.Tensor,       # [rows_padded, K] float8_e4m3fn (gather: [m, K])
    a_sf: torch.Tensor,      # [rows_padded, K/32] u8
    w_fp4: torch.Tensor,     # [E, N, K/2] u8
    w_sf: torch.Tensor,      # [E, N, K/32] u8
    expert_ids: torch.Tensor,  # [num_m_blocks] i32
    active_m: int,
    c: torch.Tensor,         # [rows_padded, N] bf16 (in/out)
    *,
    packed_route_indices: torch.Tensor | None = None,
    topk: int = 1,
    total_routes: int = 0,
    experts_per_group: bool = False,
) -> torch.Tensor:
    e, n, k_half = w_fp4.shape
    k = k_half * 2
    gather = packed_route_indices is not None
    rows_padded = c.shape[0]
    num_m_blocks = rows_padded // _BLOCK_M
    assert rows_padded % _BLOCK_M == 0
    if not experts_per_group:
        assert expert_ids.shape[0] == num_m_blocks

    b_rp, sfb = repack_w4a8_weights(w_fp4, w_sf)
    kernel = W4A8GemmKernel(
        size_n=n,
        size_k=k,
        num_experts=e,
        gather_a=gather,
        topk=topk,
        experts_per_group=experts_per_group,
    )
    grid_x = kernel.grid_x(num_m_blocks)
    if not gather:
        # Dense mode ignores the route tensor; pass the expert ids as dummy.
        packed_route_indices = expert_ids

    args = (
        to_cute_u32(a_q),
        to_cute_u32(a_sf),
        to_cute_u32(b_rp),
        to_cute_u32(sfb),
        to_cute_u32(c),
        _to_cute_i32(expert_ids),
        _to_cute_i32(packed_route_indices),
        cutlass.Int32(num_m_blocks),
        cutlass.Int32(active_m),
        cutlass.Int32(total_routes),
        cutlass.Int32(grid_x),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )
    key = (e, n, k, rows_padded, gather, topk, experts_per_group)
    compiled = _COMPILE_CACHE.get(key)
    if compiled is None:
        compiled = cute.compile(kernel, *args)
        _COMPILE_CACHE[key] = compiled
    compiled(*args)
    torch.cuda.synchronize()
    return c


def _dequant_a(a_q: torch.Tensor, a_sf: torch.Tensor) -> torch.Tensor:
    """payload * 2^(sf-127), scale byte 0 -> scale 0, f32."""
    m, k = a_q.shape
    scale = torch.where(
        a_sf == 0,
        torch.zeros((), device=a_q.device),
        torch.exp2(a_sf.to(torch.float32) - 127.0),
    )
    return (a_q.to(torch.float32).view(m, k // 32, 32) * scale.unsqueeze(-1)).view(
        m, k
    )


def _oracle(
    a_q: torch.Tensor,
    a_sf: torch.Tensor,
    w_fp4: torch.Tensor,
    w_sf: torch.Tensor,
    expert_ids: torch.Tensor,
) -> torch.Tensor:
    """f32 grouped GEMM: per m-block, A_deq @ B_deq[expert].T."""
    e, n, k_half = w_fp4.shape
    k = k_half * 2
    lut = _make_fp4_lut(a_q.device)
    w_eff = torch.stack(
        [
            _dequant_w4a8_weight_e8m0_k32(w_fp4[i], w_sf[i], n, k, lut).view(n, k)
            for i in range(e)
        ]
    )
    a_deq = _dequant_a(a_q, a_sf)
    out = torch.zeros(a_q.shape[0], n, dtype=torch.float32, device=a_q.device)
    for mb in range(expert_ids.shape[0]):
        eid = int(expert_ids[mb].item())
        if eid < 0:
            continue
        rows = slice(mb * _BLOCK_M, (mb + 1) * _BLOCK_M)
        out[rows] = a_deq[rows] @ w_eff[eid].T
    return out


def _dyadic_activations(rows: int, k: int, seed: int) -> torch.Tensor:
    """Values whose mxfp8 row-quantization round-trips exactly."""
    torch.manual_seed(seed)
    device = torch.device("cuda")
    vals = torch.tensor(
        [0.0, 0.5, 1.0, 2.0, -1.0, -0.5, 4.0, -2.0], device=device
    )
    return vals[torch.randint(0, 8, (rows, k), device=device)].to(torch.bfloat16)


def _dyadic_weights(e: int, n: int, k: int, seed: int):
    """FP4-grid payloads + power-of-two e8m0 block scales: dequant is exact."""
    torch.manual_seed(seed)
    device = torch.device("cuda")
    fp4_grid = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
         -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        device=device,
    )
    payload = fp4_grid[torch.randint(0, 15, (e, n, k), device=device)]
    w_fp4 = _pack_b(payload.view(e * n, k)).view(e, n, k // 2)
    w_sf = (
        torch.randint(-3, 4, (e, n, k // 32), device=device) + 127
    ).to(torch.uint8)
    return w_fp4, w_sf


def _case(
    num_m_blocks: int,
    expert_list: list[int],
    *,
    e: int,
    n: int,
    k: int,
    seed: int,
    active_m: int | None = None,
):
    device = torch.device("cuda")
    rows_padded = num_m_blocks * _BLOCK_M
    if active_m is None:
        active_m = rows_padded
    x = _dyadic_activations(rows_padded, k, seed)
    a_q, a_sf = mxfp8_quantize_rows(x)
    w_fp4, w_sf = _dyadic_weights(e, n, k, seed + 1)
    expert_ids = torch.tensor(expert_list, dtype=torch.int32, device=device)
    return a_q, a_sf, w_fp4, w_sf, expert_ids, active_m


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    "num_m_blocks,expert_list",
    [
        (1, [0]),                 # single m-block (half pair)
        (3, [1, 0, 1]),           # mixed experts: mismatched pairs slow path
        (4, [1, 1, 0, 0]),        # expert-uniform pairs: shared-B fast path
        (4, [0, 1, 1, -1]),       # straddling pair + padding-block pair
    ],
)
def test_w4a8_gemm_dyadic_exact(num_m_blocks: int, expert_list: list[int]) -> None:
    """Dyadic data: kernel bf16 output == once-rounded f32 oracle, atol=0."""
    require_sm120()
    e, n, k = 2, 512, 512
    a_q, a_sf, w_fp4, w_sf, expert_ids, active_m = _case(
        num_m_blocks, expert_list, e=e, n=n, k=k, seed=100 + num_m_blocks
    )
    c = torch.zeros(a_q.shape[0], n, dtype=torch.bfloat16, device=a_q.device)
    _run_gemm(a_q, a_sf, w_fp4, w_sf, expert_ids, active_m, c)
    ref = _oracle(a_q, a_sf, w_fp4, w_sf, expert_ids)
    assert ref.abs().sum().item() > 0
    torch.testing.assert_close(c, ref.to(torch.bfloat16), atol=0.0, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_w4a8_gemm_partial_last_block_masking() -> None:
    """Rows beyond active_m are never stored; valid rows stay exact."""
    require_sm120()
    e, n, k = 2, 512, 512
    num_m_blocks, active_m = 3, 40   # last block has 8 valid rows
    a_q, a_sf, w_fp4, w_sf, expert_ids, _ = _case(
        num_m_blocks, [0, 1, 0], e=e, n=n, k=k, seed=200, active_m=active_m
    )
    sentinel = 777.0
    c = torch.full(
        (a_q.shape[0], n), sentinel, dtype=torch.bfloat16, device=a_q.device
    )
    _run_gemm(a_q, a_sf, w_fp4, w_sf, expert_ids, active_m, c)
    ref = _oracle(a_q, a_sf, w_fp4, w_sf, expert_ids)
    torch.testing.assert_close(
        c[:active_m], ref[:active_m].to(torch.bfloat16), atol=0.0, rtol=0.0
    )
    assert torch.all(c[active_m:] == sentinel), "masked rows were written"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_w4a8_gemm_random_cosine() -> None:
    """Random data + _quantize_weight_mxfp4 weights vs the dequant oracle."""
    require_sm120()
    device = torch.device("cuda")
    torch.manual_seed(300)
    e, n, k = 2, 512, 512
    num_m_blocks = 3
    rows_padded = num_m_blocks * _BLOCK_M

    x = torch.randn(rows_padded, k, dtype=torch.bfloat16, device=device)
    a_q, a_sf = mxfp8_quantize_rows(x)
    w = torch.randn(e, n, k, device=device)
    packed = [_quantize_weight_mxfp4(w[i]) for i in range(e)]
    w_fp4 = torch.stack([p[0] for p in packed])
    w_sf = torch.stack([p[1] for p in packed])
    expert_ids = torch.tensor([1, 0, 1], dtype=torch.int32, device=device)

    c = torch.zeros(rows_padded, n, dtype=torch.bfloat16, device=device)
    _run_gemm(a_q, a_sf, w_fp4, w_sf, expert_ids, rows_padded, c)
    ref = _oracle(a_q, a_sf, w_fp4, w_sf, expert_ids)
    assert c.float().abs().sum().item() > 0
    cos = F.cosine_similarity(
        c.float().reshape(1, -1), ref.reshape(1, -1)
    ).item()
    assert cos > 0.9999, f"cosine {cos}"
    torch.testing.assert_close(c.float(), ref, atol=2e-1, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_w4a8_gemm_gather_mode() -> None:
    """Route-gathered A rows via the real w4a16 route packing, dyadic exact.

    Checks (a) gathered rows equal the oracle of A[idx // topk] @ B[expert].T
    at the packed positions, (b) padding slots inside routed blocks are
    stored as ZEROS, and (c) rows of -1 (absent) blocks are untouched.
    """
    require_sm120()
    from b12x.moe.fused.w4a16.route_pack import pack_topk_routes_by_expert

    device = torch.device("cuda")
    e, n, k = 4, 512, 512
    m, topk = 21, 3
    torch.manual_seed(400)

    x = _dyadic_activations(m, k, 401)
    a_q, a_sf = mxfp8_quantize_rows(x)
    w_fp4, w_sf = _dyadic_weights(e, n, k, 402)
    topk_ids = torch.stack(
        [torch.randperm(e, device=device)[:topk] for _ in range(m)]
    ).to(torch.int32)
    # Gather-mode contract: the route-index buffer must cover all staged
    # rows (num_m_blocks*16); slots beyond route_pack's written capacity are
    # prefilled once with an invalid value (>= total_routes).
    from b12x.moe.fused.w4a16.host import (
        max_packed_route_slots,
        route_pack_numel_capacity,
    )

    total_routes = m * topk
    cap_slots = max_packed_route_slots(
        route_pack_numel_capacity(total_routes, topk=topk), _BLOCK_M, e
    )
    rows_padded = ((cap_slots + _BLOCK_M - 1) // _BLOCK_M) * _BLOCK_M
    pri_buf = torch.full(
        (rows_padded,), torch.iinfo(torch.int32).max,
        dtype=torch.int32, device=device,
    )
    pri, beids, count = pack_topk_routes_by_expert(
        topk_ids, _BLOCK_M, e, packed_route_indices=pri_buf
    )
    num_m_blocks = int(beids.numel())
    assert num_m_blocks * _BLOCK_M == rows_padded

    sentinel = 777.0
    c = torch.full((rows_padded, n), sentinel, dtype=torch.bfloat16, device=device)
    _run_gemm(
        a_q, a_sf, w_fp4, w_sf, beids, rows_padded, c,
        packed_route_indices=pri_buf, topk=topk, total_routes=total_routes,
    )

    # Oracle at packed positions.
    lut = _make_fp4_lut(device)
    w_eff = torch.stack(
        [
            _dequant_w4a8_weight_e8m0_k32(w_fp4[i], w_sf[i], n, k, lut).view(n, k)
            for i in range(e)
        ]
    )
    a_deq = _dequant_a(a_q, a_sf)
    ref = torch.full((rows_padded, n), sentinel, dtype=torch.float32, device=device)
    pri_h = pri_buf.cpu()
    beids_h = beids.cpu()
    n_pad_slots = 0
    for s in range(rows_padded):
        eid = int(beids_h[s // _BLOCK_M].item())
        if eid < 0:
            continue
        idx = int(pri_h[s].item())
        if idx < total_routes:
            ref[s] = a_deq[idx // topk] @ w_eff[eid].T
        else:
            ref[s] = 0.0
            n_pad_slots += 1
    assert n_pad_slots > 0, "test must exercise padding slots"
    assert int((beids_h >= 0).sum()) * _BLOCK_M >= total_routes
    live = int(count.item())
    assert ref[:live].abs().sum().item() > 0
    torch.testing.assert_close(c, ref.to(torch.bfloat16), atol=0.0, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_w4a8_gemm_gather_experts_per_group() -> None:
    """Gather mode + experts_per_group with the w4a8 group-padded route pack:
    every group uniform, dyadic exact, group-padding slots zeroed."""
    require_sm120()
    from b12x.moe.fused.w4a8.route import pack_routes_w4a8, w4a8_route_capacity

    device = torch.device("cuda")
    e, n, k = 4, 512, 512
    m, topk = 21, 3
    group_rows = 48
    torch.manual_seed(500)

    x = _dyadic_activations(m, k, 501)
    a_q, a_sf = mxfp8_quantize_rows(x)
    w_fp4, w_sf = _dyadic_weights(e, n, k, 502)
    topk_ids = torch.stack(
        [torch.randperm(e, device=device)[:topk] for _ in range(m)]
    ).to(torch.int32)
    total_routes = m * topk
    cap_rows, cap_groups = w4a8_route_capacity(total_routes, e, group_rows)
    pri_buf = torch.full(
        (cap_rows,), torch.iinfo(torch.int32).max,
        dtype=torch.int32, device=device,
    )
    beids = torch.empty(cap_groups, dtype=torch.int32, device=device)
    count = torch.zeros(1, dtype=torch.int32, device=device)
    eoff = torch.empty(e + 1, dtype=torch.int32, device=device)
    ecnt = torch.zeros(e, dtype=torch.int32, device=device)
    pack_routes_w4a8(
        topk_ids, e, group_rows,
        packed_route_indices=pri_buf, block_expert_ids=beids,
        packed_route_count=count, expert_offsets=eoff, expert_counts=ecnt,
    )

    sentinel = 777.0
    c = torch.full((cap_rows, n), sentinel, dtype=torch.bfloat16, device=device)
    _run_gemm(
        a_q, a_sf, w_fp4, w_sf, beids, cap_rows, c,
        packed_route_indices=pri_buf, topk=topk, total_routes=total_routes,
        experts_per_group=True,
    )

    lut = _make_fp4_lut(device)
    w_eff = torch.stack(
        [
            _dequant_w4a8_weight_e8m0_k32(w_fp4[i], w_sf[i], n, k, lut).view(n, k)
            for i in range(e)
        ]
    )
    a_deq = _dequant_a(a_q, a_sf)
    ref = torch.full((cap_rows, n), sentinel, dtype=torch.float32, device=device)
    pri_h = pri_buf.cpu()
    beids_h = beids.cpu()
    for s in range(cap_rows):
        eid = int(beids_h[s // group_rows].item())
        if eid < 0:
            continue
        idx = int(pri_h[s].item())
        if idx < total_routes:
            ref[s] = a_deq[idx // topk] @ w_eff[eid].T
        else:
            ref[s] = 0.0
    live = int(count.item())
    assert ref[:live].abs().sum().item() > 0
    torch.testing.assert_close(c, ref.to(torch.bfloat16), atol=0.0, rtol=0.0)
