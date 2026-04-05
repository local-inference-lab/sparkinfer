"""Small reusable ops for the model forward pass.

RMSNorm, partial RoPE, FP8 KV cache writes. These are not
model-specific — any transformer recipe can use them.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from b12x.profiling import record_function


def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    gemma_style: bool = False,
) -> torch.Tensor:
    """RMSNorm: x / rms(x) * scale.

    If gemma_style, scale = (1 + weight) (Gemma/Qwen3.5 convention where
    checkpoint stores the offset from 1.0).
    """
    with record_function("model.rms_norm"):
        dtype = x.dtype
        with record_function("model.rms_norm.cast_fp32"):
            x = x.float()
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        scale = (1.0 + weight) if gemma_style else weight
        with record_function("model.rms_norm.cast_output"):
            return (x * rms).to(dtype) * scale


class RMSNorm(torch.nn.Module):
    """Standard RMSNorm: x / rms(x) * weight."""

    def __init__(self, weight: torch.Tensor, eps: float = 1e-6):
        super().__init__()
        self.register_buffer("weight", weight)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm(x, self.weight, self.eps)


class GemmaRMSNorm(torch.nn.Module):
    """Gemma-style RMSNorm: x / rms(x) * (1 + weight)."""

    def __init__(self, weight: torch.Tensor, eps: float = 1e-6):
        super().__init__()
        self.register_buffer("weight", weight)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm(x, self.weight, self.eps, gemma_style=True)


def make_norm(weight: torch.Tensor, eps: float = 1e-6, gemma_style: bool = False) -> torch.nn.Module:
    """Factory: returns the right RMSNorm variant."""
    return GemmaRMSNorm(weight, eps) if gemma_style else RMSNorm(weight, eps)


def precompute_rope_freqs(
    head_dim: int,
    rotary_dim: int | None = None,
    max_seq_len: int = 32768,
    base: float = 10000.0,
    device: torch.device | str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos/sin tables for RoPE.

    Returns (cos, sin) each [max_seq_len, rotary_dim].
    For partial RoPE (rotary_dim < head_dim), only the first rotary_dim
    dimensions of Q/K are rotated.
    """
    if rotary_dim is None:
        rotary_dim = head_dim
    inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, device=device).float() / rotary_dim))
    t = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)  # [max_seq_len, rotary_dim // 2].
    emb = torch.cat([freqs, freqs], dim=-1)  # [max_seq_len, rotary_dim].
    cos = torch.cos(emb).to(torch.bfloat16)
    sin = torch.sin(emb).to(torch.bfloat16)
    return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_partial_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to the first *rotary_dim* dims, pass-through the rest.

    q: [total_q, num_q_heads, head_dim].
    k: [total_q, num_kv_heads, head_dim].
    cos, sin: [total_q, rotary_dim] (already indexed by position).

    Uses the standard rotate_half formulation matching HuggingFace.
    """
    # Expand cos/sin for head broadcasting: [total_q, 1, rotary_dim].
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    q_rot = q[..., :rotary_dim]
    q_pass = q[..., rotary_dim:]
    q_rot = (q_rot * cos) + (_rotate_half(q_rot) * sin)

    k_rot = k[..., :rotary_dim]
    k_pass = k[..., rotary_dim:]
    k_rot = (k_rot * cos) + (_rotate_half(k_rot) * sin)

    q = torch.cat([q_rot, q_pass], dim=-1) if q_pass.shape[-1] > 0 else q_rot
    k = torch.cat([k_rot, k_pass], dim=-1) if k_pass.shape[-1] > 0 else k_rot
    return q, k


def write_kv_to_cache(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    k_descale: torch.Tensor | None = None,
    v_descale: torch.Tensor | None = None,
) -> None:
    """Scatter K/V tokens into the paged cache.

    k, v: [total_q, kv_heads, head_dim] in BF16.
    k_cache, v_cache: [num_pages, page_size, kv_heads, head_dim].
    page_table: [batch, max_pages] int32.
    cache_seqlens: [batch] int32 — length BEFORE this write.
    cu_seqlens_q: [batch + 1] int32.
    k_descale, v_descale: [batch, kv_heads] float32 — written with per-head
        scale factors when cache is FP8.
    """
    page_size = k_cache.shape[1]
    batch = cache_seqlens.shape[0]
    total_q = k.shape[0]

    if total_q == batch:
        # Decode fast path: one token per request.
        cache_pos = cache_seqlens.long()
        page_indices = page_table[
            torch.arange(batch, device=k.device), cache_pos // page_size
        ]
        slot_indices = cache_pos % page_size
    elif batch == 1:
        # Single-request extend (chunked prefill). Graph-safe: no repeat_interleave.
        offsets = torch.arange(total_q, device=k.device)
        cache_pos = cache_seqlens[0].long() + offsets
        page_indices = page_table[0, (cache_pos // page_size).long()]
        slot_indices = cache_pos % page_size
    else:
        # Multi-request extend: variable tokens per request.
        q_lens = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
        token_indices = torch.arange(total_q, device=k.device)
        batch_ids = torch.repeat_interleave(
            torch.arange(batch, device=k.device), q_lens.long()
        )
        offsets_in_req = token_indices - cu_seqlens_q[batch_ids]
        cache_pos = cache_seqlens[batch_ids] + offsets_in_req
        page_indices = page_table[batch_ids, (cache_pos // page_size).long()]
        slot_indices = cache_pos % page_size

    k_cache[page_indices.long(), slot_indices.long()] = k.to(k_cache.dtype)
    v_cache[page_indices.long(), slot_indices.long()] = v.to(v_cache.dtype)
