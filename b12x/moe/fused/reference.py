from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from b12x.cute.fp4 import fp4_quantize_values_torch


@dataclass(frozen=True)
class OracleMetrics:
    max_abs: float
    rmse: float
    mean_abs: float
    cos: float


def compare_to_reference(actual: torch.Tensor, reference: torch.Tensor) -> OracleMetrics:
    actual_fp32 = actual.float()
    reference_fp32 = reference.float()
    diff = actual_fp32 - reference_fp32
    cos = F.cosine_similarity(
        actual_fp32.reshape(actual_fp32.shape[0], -1),
        reference_fp32.reshape(reference_fp32.shape[0], -1),
        dim=1,
    ).mean().item()
    return OracleMetrics(
        max_abs=diff.abs().max().item(),
        rmse=diff.square().mean().sqrt().item(),
        mean_abs=diff.abs().mean().item(),
        cos=cos,
    )


def _routed_expert_tiles(topk_ids: torch.Tensor, topk_weights: torch.Tensor, *, tile_rows: int = 128):
    expert_rows: dict[int, list[tuple[int, float]]] = {}
    m, top_k = topk_ids.shape
    for t in range(m):
        for k_idx in range(top_k):
            eid = int(topk_ids[t, k_idx].item())
            router_w = float(topk_weights[t, k_idx].item())
            expert_rows.setdefault(eid, []).append((t, router_w))
    for eid, rows in expert_rows.items():
        for tile_start in range(0, len(rows), tile_rows):
            yield eid, rows[tile_start : tile_start + tile_rows]


def unswizzle_block_scale(swizzled_scale: torch.Tensor, rows: int, cols_blocks: int) -> torch.Tensor:
    cols_padded = ((cols_blocks + 3) // 4) * 4
    rows_padded = ((rows + 127) // 128) * 128
    unswizzled = swizzled_scale.view(torch.float8_e4m3fn).reshape(
        rows_padded // 128, cols_padded // 4, 32, 4, 4,
    )
    unswizzled = unswizzled.permute(0, 3, 2, 1, 4).contiguous()
    unswizzled = unswizzled.reshape(rows_padded, cols_padded)
    return unswizzled[:rows, :cols_blocks].to(torch.float32)


def _make_fp4_lut(device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
            -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
        ],
        dtype=torch.float32,
        device=device,
    )


def _dequant_fp4(packed_u8: torch.Tensor, rows: int, cols: int, fp4_lut: torch.Tensor) -> torch.Tensor:
    lo = (packed_u8 & 0x0F).to(torch.int64)
    hi = ((packed_u8 >> 4) & 0x0F).to(torch.int64)
    return torch.stack([fp4_lut[lo], fp4_lut[hi]], dim=-1).reshape(rows, cols)


def _apply_block_scales(raw: torch.Tensor, sf_f32: torch.Tensor, rows: int, cols: int, block_size: int) -> torch.Tensor:
    n_blocks = cols // block_size
    sf = sf_f32[:rows, :n_blocks]
    return raw * sf.unsqueeze(-1).expand(rows, n_blocks, block_size).reshape(rows, cols)


def _quantize_vec_to_fp4_dequant(
    vals_f32: torch.Tensor,
    global_scale: float,
    *,
    block_size: int,
    fp8_e4m3_max: float,
) -> torch.Tensor:
    cols = vals_f32.shape[0]
    n_blocks = cols // block_size
    blocked = vals_f32.reshape(n_blocks, block_size)
    block_max = blocked.abs().amax(dim=-1)

    raw_scale = (block_max / (6.0 * global_scale)).clamp(max=fp8_e4m3_max)
    sf_e4m3 = raw_scale.to(torch.float8_e4m3fn).to(torch.float32)

    sf_times_gs = sf_e4m3.unsqueeze(-1).expand(n_blocks, block_size).reshape(cols) * global_scale
    scaled = vals_f32 / sf_times_gs.clamp(min=1e-30)
    quant = fp4_quantize_values_torch(scaled)
    sf_only = sf_e4m3.unsqueeze(-1).expand(n_blocks, block_size).reshape(cols)
    return quant * sf_only


def _effective_scale(scale: float, *, input_scales_are_reciprocal: bool) -> float:
    if not input_scales_are_reciprocal:
        return scale
    if scale == 0.0:
        return 0.0
    return 1.0 / scale


def moe_reference_f32(
    x: torch.Tensor,
    w1_fp4: torch.Tensor,
    w1_blockscale: torch.Tensor,
    w1_alphas: torch.Tensor,
    w2_fp4: torch.Tensor,
    w2_blockscale: torch.Tensor,
    w2_alphas: torch.Tensor,
    a1_gscale: torch.Tensor,
    a2_gscale: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    E: int,
    K: int,
    I_tp: int,
    *,
    input_scales_are_reciprocal: bool = False,
    fc2_tile_amax: bool = False,
) -> torch.Tensor:
    del E
    del fc2_tile_amax
    block_size = 16
    fp8_e4m3_max = float(torch.finfo(torch.float8_e4m3fn).max)
    fp4_lut = _make_fp4_lut(x.device)

    device = x.device
    m = x.shape[0]
    top_k = topk_ids.shape[1]
    output = torch.zeros(m, K, dtype=torch.float32, device=device)

    for t in range(m):
        x_f32 = x[t].float()
        for k_idx in range(top_k):
            eid = int(topk_ids[t, k_idx].item())
            router_w = float(topk_weights[t, k_idx].item())
            alpha_fc1 = float(w1_alphas[eid].item())
            alpha_fc2 = float(w2_alphas[eid].item())

            gs_fc1 = _effective_scale(
                float(a1_gscale[eid].item()) if a1_gscale.numel() > 1 else float(a1_gscale.item())
                , input_scales_are_reciprocal=input_scales_are_reciprocal
            )
            gs_fc2 = _effective_scale(
                float(a2_gscale[eid].item()) if a2_gscale.numel() > 1 else float(a2_gscale.item())
                , input_scales_are_reciprocal=input_scales_are_reciprocal
            )

            x_dequant = _quantize_vec_to_fp4_dequant(
                x_f32, gs_fc1, block_size=block_size, fp8_e4m3_max=fp8_e4m3_max
            )

            w13_sf = unswizzle_block_scale(w1_blockscale[eid], 2 * I_tp, K // block_size)
            w2_sf = unswizzle_block_scale(w2_blockscale[eid], K, I_tp // block_size)

            up_dequant = _apply_block_scales(
                _dequant_fp4(w1_fp4[eid, :I_tp], I_tp, K, fp4_lut), w13_sf[:I_tp], I_tp, K, block_size,
            )
            gate_dequant = _apply_block_scales(
                _dequant_fp4(w1_fp4[eid, I_tp:], I_tp, K, fp4_lut), w13_sf[I_tp:], I_tp, K, block_size,
            )

            gate_out = (gate_dequant @ x_dequant) * alpha_fc1
            up_out = (up_dequant @ x_dequant) * alpha_fc1
            intermediate = torch.sigmoid(gate_out) * gate_out * up_out

            int_dequant = _quantize_vec_to_fp4_dequant(
                intermediate, gs_fc2, block_size=block_size, fp8_e4m3_max=fp8_e4m3_max
            )
            down_dequant = _apply_block_scales(
                _dequant_fp4(w2_fp4[eid], K, I_tp, fp4_lut), w2_sf, K, I_tp, block_size,
            )
            down_out = (down_dequant @ int_dequant) * alpha_fc2
            output[t] += router_w * down_out

    return output.to(torch.bfloat16)


def moe_reference_nvfp4(
    x: torch.Tensor,
    w1_fp4: torch.Tensor,
    w1_blockscale: torch.Tensor,
    w1_alphas: torch.Tensor,
    w2_fp4: torch.Tensor,
    w2_blockscale: torch.Tensor,
    w2_alphas: torch.Tensor,
    a1_gscale: torch.Tensor,
    a2_gscale: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    E: int,
    K: int,
    I_tp: int,
    *,
    input_scales_are_reciprocal: bool = False,
    fc2_tile_amax: bool = False,
) -> torch.Tensor:
    del fc2_tile_amax
    block_size = 16
    fp8_e4m3_max = float(torch.finfo(torch.float8_e4m3fn).max)
    fp4_lut = _make_fp4_lut(x.device)

    m = x.shape[0]
    top_k = topk_ids.shape[1]
    output = torch.zeros(m, K, dtype=torch.bfloat16, device=x.device)
    contribs: list[list[tuple[int, torch.Tensor]]] = [[] for _ in range(E)]

    for t in range(m):
        x_f32 = x[t].float()
        for k_idx in range(top_k):
            eid = int(topk_ids[t, k_idx].item())
            router_w = float(topk_weights[t, k_idx].item())
            alpha_fc1 = float(w1_alphas[eid].item())
            alpha_fc2 = float(w2_alphas[eid].item())
            gs_fc1 = _effective_scale(
                float(a1_gscale[eid].item()) if a1_gscale.numel() > 1 else float(a1_gscale.item())
                , input_scales_are_reciprocal=input_scales_are_reciprocal
            )
            gs_fc2 = _effective_scale(
                float(a2_gscale[eid].item()) if a2_gscale.numel() > 1 else float(a2_gscale.item())
                , input_scales_are_reciprocal=input_scales_are_reciprocal
            )

            x_dequant = _quantize_vec_to_fp4_dequant(
                x_f32, gs_fc1, block_size=block_size, fp8_e4m3_max=fp8_e4m3_max
            )

            w13_sf = unswizzle_block_scale(w1_blockscale[eid], 2 * I_tp, K // block_size)
            w2_sf = unswizzle_block_scale(w2_blockscale[eid], K, I_tp // block_size)

            up_dequant = _apply_block_scales(
                _dequant_fp4(w1_fp4[eid, :I_tp], I_tp, K, fp4_lut), w13_sf[:I_tp], I_tp, K, block_size,
            )
            gate_dequant = _apply_block_scales(
                _dequant_fp4(w1_fp4[eid, I_tp:], I_tp, K, fp4_lut), w13_sf[I_tp:], I_tp, K, block_size,
            )

            gate_out = (gate_dequant @ x_dequant) * alpha_fc1
            up_out = (up_dequant @ x_dequant) * alpha_fc1
            intermediate = (torch.sigmoid(gate_out) * gate_out * up_out).to(torch.bfloat16).float()

            int_dequant = _quantize_vec_to_fp4_dequant(
                intermediate, gs_fc2, block_size=block_size, fp8_e4m3_max=fp8_e4m3_max
            )
            down_dequant = _apply_block_scales(
                _dequant_fp4(w2_fp4[eid], K, I_tp, fp4_lut), w2_sf, K, I_tp, block_size,
            )
            down_out = ((down_dequant @ int_dequant) * alpha_fc2).to(torch.bfloat16)
            contribs[eid].append((t, (router_w * down_out.float()).to(torch.bfloat16)))

    for eid in range(E):
        for t, contrib in contribs[eid]:
            output[t] = (output[t].float() + contrib.float()).to(torch.bfloat16)

    return output


def moe_reference_nvfp4_fc2_tiled(
    x: torch.Tensor,
    w1_fp4: torch.Tensor,
    w1_blockscale: torch.Tensor,
    w1_alphas: torch.Tensor,
    w2_fp4: torch.Tensor,
    w2_blockscale: torch.Tensor,
    w2_alphas: torch.Tensor,
    a1_gscale: torch.Tensor,
    a2_gscale: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    E: int,
    K: int,
    I_tp: int,
    *,
    input_scales_are_reciprocal: bool = False,
    fc2_tile_amax: bool = False,
) -> torch.Tensor:
    return moe_reference_nvfp4_fc1_fc2_tiled(
        x,
        w1_fp4,
        w1_blockscale,
        w1_alphas,
        w2_fp4,
        w2_blockscale,
        w2_alphas,
        a1_gscale,
        a2_gscale,
        topk_ids,
        topk_weights,
        E,
        K,
        I_tp,
        input_scales_are_reciprocal=input_scales_are_reciprocal,
        fc1_tile_amax=False,
        fc2_tile_amax=fc2_tile_amax,
    )


def moe_reference_nvfp4_fc1_fc2_tiled(
    x: torch.Tensor,
    w1_fp4: torch.Tensor,
    w1_blockscale: torch.Tensor,
    w1_alphas: torch.Tensor,
    w2_fp4: torch.Tensor,
    w2_blockscale: torch.Tensor,
    w2_alphas: torch.Tensor,
    a1_gscale: torch.Tensor,
    a2_gscale: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    E: int,
    K: int,
    I_tp: int,
    *,
    input_scales_are_reciprocal: bool = False,
    fc1_tile_amax: bool = False,
    fc2_tile_amax: bool = False,
) -> torch.Tensor:
    block_size = 16
    fp8_e4m3_max = float(torch.finfo(torch.float8_e4m3fn).max)
    fp4_lut = _make_fp4_lut(x.device)

    expert_cache: list[tuple[float, float, float, float, torch.Tensor, torch.Tensor, torch.Tensor] | None] = [None] * E
    contribs: list[list[tuple[int, float, torch.Tensor]]] = [[] for _ in range(E)]
    output = torch.zeros(x.shape[0], K, dtype=torch.bfloat16, device=x.device)

    for t in range(x.shape[0]):
        x_f32 = x[t].float()
        for k_idx in range(topk_ids.shape[1]):
            eid = int(topk_ids[t, k_idx].item())
            router_w = float(topk_weights[t, k_idx].item())
            cached = expert_cache[eid]
            if cached is None:
                alpha_fc1 = float(w1_alphas[eid].item())
                alpha_fc2 = float(w2_alphas[eid].item())
                gs_fc1 = _effective_scale(
                    float(a1_gscale[eid].item()) if a1_gscale.numel() > 1 else float(a1_gscale.item()),
                    input_scales_are_reciprocal=input_scales_are_reciprocal,
                )
                gs_fc2 = _effective_scale(
                    float(a2_gscale[eid].item()) if a2_gscale.numel() > 1 else float(a2_gscale.item()),
                    input_scales_are_reciprocal=input_scales_are_reciprocal,
                )
                w13_sf = unswizzle_block_scale(w1_blockscale[eid], 2 * I_tp, K // block_size)
                w2_sf = unswizzle_block_scale(w2_blockscale[eid], K, I_tp // block_size)
                up_dequant = _apply_block_scales(
                    _dequant_fp4(w1_fp4[eid, :I_tp], I_tp, K, fp4_lut), w13_sf[:I_tp], I_tp, K, block_size,
                )
                gate_dequant = _apply_block_scales(
                    _dequant_fp4(w1_fp4[eid, I_tp:], I_tp, K, fp4_lut), w13_sf[I_tp:], I_tp, K, block_size,
                )
                down_dequant = _apply_block_scales(
                    _dequant_fp4(w2_fp4[eid], K, I_tp, fp4_lut), w2_sf, K, I_tp, block_size,
                )
                cached = (alpha_fc1, alpha_fc2, gs_fc1, gs_fc2, up_dequant, gate_dequant, down_dequant)
                expert_cache[eid] = cached

            contribs[eid].append((t, router_w, x_f32))

    for eid in range(E):
        entries = contribs[eid]
        if not entries:
            continue
        alpha_fc1, alpha_fc2, gs_fc1, gs_fc2, up_dequant, gate_dequant, down_dequant = expert_cache[eid]
        for tile_start in range(0, len(entries), 128):
            tile_entries = entries[tile_start : tile_start + 128]
            gs_fc1_quant = gs_fc1
            alpha_fc1_quant = alpha_fc1
            if fc1_tile_amax:
                tile_amax = max(float(x_row.abs().amax().item()) for _, _, x_row in tile_entries)
                if tile_amax > 0.0 and gs_fc1 != 0.0:
                    gs_fc1_quant = tile_amax / (6.0 * fp8_e4m3_max)
                    alpha_fc1_quant = alpha_fc1 * (gs_fc1_quant / gs_fc1)

            tile_intermediates: list[tuple[int, float, torch.Tensor]] = []
            for t, router_w, x_f32 in tile_entries:
                x_dequant = _quantize_vec_to_fp4_dequant(
                    x_f32, gs_fc1_quant, block_size=block_size, fp8_e4m3_max=fp8_e4m3_max
                )
                gate_out = (gate_dequant @ x_dequant) * alpha_fc1_quant
                up_out = (up_dequant @ x_dequant) * alpha_fc1_quant
                intermediate = (torch.sigmoid(gate_out) * gate_out * up_out).to(torch.bfloat16).float()
                tile_intermediates.append((t, router_w, intermediate))

            gs_fc2_quant = gs_fc2
            alpha_fc2_quant = alpha_fc2
            if fc2_tile_amax:
                tile_amax = max(float(intermediate.abs().amax().item()) for _, _, intermediate in tile_intermediates)
                if tile_amax > 0.0 and gs_fc2 != 0.0:
                    gs_fc2_quant = tile_amax / (6.0 * fp8_e4m3_max)
                    alpha_fc2_quant = alpha_fc2 * (gs_fc2_quant / gs_fc2)
            for t, router_w, intermediate in tile_intermediates:
                int_dequant = _quantize_vec_to_fp4_dequant(
                    intermediate, gs_fc2_quant, block_size=block_size, fp8_e4m3_max=fp8_e4m3_max
                )
                down_out = ((down_dequant @ int_dequant) * alpha_fc2_quant).to(torch.bfloat16)
                contrib = (router_w * down_out.float()).to(torch.bfloat16)
                output[t] = (output[t].float() + contrib.float()).to(torch.bfloat16)

    return output


def moe_reference_fp32_pure(
    x: torch.Tensor,
    w1_fp4: torch.Tensor,
    w1_blockscale: torch.Tensor,
    w1_alphas: torch.Tensor,
    w2_fp4: torch.Tensor,
    w2_blockscale: torch.Tensor,
    w2_alphas: torch.Tensor,
    a1_gscale: torch.Tensor,
    a2_gscale: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    E: int,
    K: int,
    I_tp: int,
    *,
    input_scales_are_reciprocal: bool = False,
    fc2_tile_amax: bool = False,
) -> torch.Tensor:
    del fc2_tile_amax
    block_size = 16
    fp4_lut = _make_fp4_lut(x.device)

    expert_cache: list[tuple[float, float, torch.Tensor, torch.Tensor, torch.Tensor] | None] = [None] * E
    output = torch.zeros(x.shape[0], K, dtype=torch.float32, device=x.device)

    for t in range(x.shape[0]):
        x_f32 = x[t].float()
        for k_idx in range(topk_ids.shape[1]):
            eid = int(topk_ids[t, k_idx].item())
            router_w = float(topk_weights[t, k_idx].item())
            cached = expert_cache[eid]
            if cached is None:
                gs_fc1 = _effective_scale(
                    float(a1_gscale[eid].item()) if a1_gscale.numel() > 1 else float(a1_gscale.item()),
                    input_scales_are_reciprocal=input_scales_are_reciprocal,
                )
                gs_fc2 = _effective_scale(
                    float(a2_gscale[eid].item()) if a2_gscale.numel() > 1 else float(a2_gscale.item()),
                    input_scales_are_reciprocal=input_scales_are_reciprocal,
                )
                alpha_fc1 = float(w1_alphas[eid].item())
                alpha_fc2 = float(w2_alphas[eid].item())
                if gs_fc1 != 0.0:
                    alpha_fc1 /= gs_fc1
                if gs_fc2 != 0.0:
                    alpha_fc2 /= gs_fc2
                w13_sf = unswizzle_block_scale(w1_blockscale[eid], 2 * I_tp, K // block_size)
                w2_sf = unswizzle_block_scale(w2_blockscale[eid], K, I_tp // block_size)
                up_dequant = _apply_block_scales(
                    _dequant_fp4(w1_fp4[eid, :I_tp], I_tp, K, fp4_lut), w13_sf[:I_tp], I_tp, K, block_size,
                )
                gate_dequant = _apply_block_scales(
                    _dequant_fp4(w1_fp4[eid, I_tp:], I_tp, K, fp4_lut), w13_sf[I_tp:], I_tp, K, block_size,
                )
                down_dequant = _apply_block_scales(
                    _dequant_fp4(w2_fp4[eid], K, I_tp, fp4_lut), w2_sf, K, I_tp, block_size,
                )
                cached = (alpha_fc1, alpha_fc2, up_dequant, gate_dequant, down_dequant)
                expert_cache[eid] = cached

            alpha_fc1, alpha_fc2, up_dequant, gate_dequant, down_dequant = cached
            gate_out = (gate_dequant @ x_f32) * alpha_fc1
            up_out = (up_dequant @ x_f32) * alpha_fc1
            intermediate = torch.sigmoid(gate_out) * gate_out * up_out
            down_out = (down_dequant @ intermediate) * alpha_fc2
            output[t] += router_w * down_out

    return output
