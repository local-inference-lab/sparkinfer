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


def unswizzle_block_scale(swizzled_scale: torch.Tensor, rows: int, cols_blocks: int) -> torch.Tensor:
    cols_padded = ((cols_blocks + 3) // 4) * 4
    rows_padded = ((rows + 127) // 128) * 128
    unswizzled = swizzled_scale.view(torch.float8_e4m3fn).reshape(
        rows_padded // 128, cols_padded // 4, 32, 4, 4,
    )
    unswizzled = unswizzled.permute(0, 3, 2, 1, 4).contiguous()
    unswizzled = unswizzled.reshape(rows_padded, cols_padded)
    return unswizzled[:rows, :cols_blocks].to(torch.float32)


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
    activation: str = "silu",
) -> torch.Tensor:
    del E
    is_gated = activation != "relu2"
    block_size = 16
    fp8_e4m3_max = float(torch.finfo(torch.float8_e4m3fn).max)

    fp4_lut = torch.tensor(
        [
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
            -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
        ],
        dtype=torch.float32,
        device=x.device,
    )
    def dequant_fp4(packed_u8: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
        lo = (packed_u8 & 0x0F).to(torch.int64)
        hi = ((packed_u8 >> 4) & 0x0F).to(torch.int64)
        return torch.stack([fp4_lut[lo], fp4_lut[hi]], dim=-1).reshape(rows, cols)

    def apply_block_scales(raw: torch.Tensor, sf_f32: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
        n_blocks = cols // block_size
        sf = sf_f32[:rows, :n_blocks]
        return raw * sf.unsqueeze(-1).expand(rows, n_blocks, block_size).reshape(rows, cols)

    def quantize_vec_to_fp4_dequant(vals_f32: torch.Tensor, global_scale: float) -> torch.Tensor:
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

            gs_fc1 = float(a1_gscale[eid].item()) if a1_gscale.numel() > 1 else float(a1_gscale.item())
            gs_fc2 = float(a2_gscale[eid].item()) if a2_gscale.numel() > 1 else float(a2_gscale.item())

            x_dequant = quantize_vec_to_fp4_dequant(x_f32, gs_fc1)

            w2_sf = unswizzle_block_scale(w2_blockscale[eid], K, I_tp // block_size)

            if is_gated:
                w13_sf = unswizzle_block_scale(w1_blockscale[eid], 2 * I_tp, K // block_size)
                up_dequant = apply_block_scales(
                    dequant_fp4(w1_fp4[eid, :I_tp], I_tp, K), w13_sf[:I_tp], I_tp, K,
                )
                gate_dequant = apply_block_scales(
                    dequant_fp4(w1_fp4[eid, I_tp:], I_tp, K), w13_sf[I_tp:], I_tp, K,
                )
                gate_out = (gate_dequant @ x_dequant) * alpha_fc1
                up_out = (up_dequant @ x_dequant) * alpha_fc1
                intermediate = torch.sigmoid(gate_out) * gate_out * up_out
            else:
                w1_sf = unswizzle_block_scale(w1_blockscale[eid], I_tp, K // block_size)
                fc1_dequant = apply_block_scales(
                    dequant_fp4(w1_fp4[eid, :I_tp], I_tp, K), w1_sf[:I_tp], I_tp, K,
                )
                fc1_out = (fc1_dequant @ x_dequant) * alpha_fc1
                intermediate = torch.square(torch.relu(fc1_out))

            int_dequant = quantize_vec_to_fp4_dequant(intermediate, gs_fc2)
            down_dequant = apply_block_scales(
                dequant_fp4(w2_fp4[eid], K, I_tp), w2_sf, K, I_tp,
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
    activation: str = "silu",
) -> torch.Tensor:
    is_gated = activation != "relu2"
    block_size = 16
    fp8_e4m3_max = float(torch.finfo(torch.float8_e4m3fn).max)

    fp4_lut = torch.tensor(
        [
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
            -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
        ],
        dtype=torch.float32,
        device=x.device,
    )
    def dequant_fp4(packed_u8: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
        lo = (packed_u8 & 0x0F).to(torch.int64)
        hi = ((packed_u8 >> 4) & 0x0F).to(torch.int64)
        return torch.stack([fp4_lut[lo], fp4_lut[hi]], dim=-1).reshape(rows, cols)

    def apply_block_scales(raw: torch.Tensor, sf_f32: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
        n_blocks = cols // block_size
        sf = sf_f32[:rows, :n_blocks]
        return raw * sf.unsqueeze(-1).expand(rows, n_blocks, block_size).reshape(rows, cols)

    def quantize_vec_to_fp4_dequant(vals_f32: torch.Tensor, global_scale: float) -> torch.Tensor:
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
            gs_fc1 = float(a1_gscale[eid].item()) if a1_gscale.numel() > 1 else float(a1_gscale.item())
            gs_fc2 = float(a2_gscale[eid].item()) if a2_gscale.numel() > 1 else float(a2_gscale.item())

            x_dequant = quantize_vec_to_fp4_dequant(x_f32, gs_fc1)

            w2_sf = unswizzle_block_scale(w2_blockscale[eid], K, I_tp // block_size)

            if is_gated:
                w13_sf = unswizzle_block_scale(w1_blockscale[eid], 2 * I_tp, K // block_size)
                up_dequant = apply_block_scales(
                    dequant_fp4(w1_fp4[eid, :I_tp], I_tp, K), w13_sf[:I_tp], I_tp, K,
                )
                gate_dequant = apply_block_scales(
                    dequant_fp4(w1_fp4[eid, I_tp:], I_tp, K), w13_sf[I_tp:], I_tp, K,
                )
                gate_out = (gate_dequant @ x_dequant) * alpha_fc1
                up_out = (up_dequant @ x_dequant) * alpha_fc1
                intermediate = (torch.sigmoid(gate_out) * gate_out * up_out).to(torch.bfloat16).float()
            else:
                w1_sf = unswizzle_block_scale(w1_blockscale[eid], I_tp, K // block_size)
                fc1_dequant = apply_block_scales(
                    dequant_fp4(w1_fp4[eid, :I_tp], I_tp, K), w1_sf[:I_tp], I_tp, K,
                )
                fc1_out = (fc1_dequant @ x_dequant) * alpha_fc1
                intermediate = torch.square(torch.relu(fc1_out)).to(torch.bfloat16).float()

            int_dequant = quantize_vec_to_fp4_dequant(intermediate, gs_fc2)
            down_dequant = apply_block_scales(
                dequant_fp4(w2_fp4[eid], K, I_tp), w2_sf, K, I_tp,
            )
            down_out = ((down_dequant @ int_dequant) * alpha_fc2).to(torch.bfloat16)
            contribs[eid].append((t, (router_w * down_out.float()).to(torch.bfloat16)))

    for eid in range(E):
        for t, contrib in contribs[eid]:
            output[t] = (output[t].float() + contrib.float()).to(torch.bfloat16)

    return output
