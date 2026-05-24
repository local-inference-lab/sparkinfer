"""Reference attention helpers for b12x attention correctness checks."""

from __future__ import annotations

import torch


def _causal_mask_right_aligned(
    seqlen_q: int,
    seqlen_k: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    q_idx = torch.arange(seqlen_q, device=device, dtype=torch.int32).view(seqlen_q, 1)
    k_idx = torch.arange(seqlen_k, device=device, dtype=torch.int32).view(1, seqlen_k)
    return k_idx > (q_idx + seqlen_k - seqlen_q)


def attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    softmax_scale: float | None = None,
    causal: bool = True,
    window_left: int = -1,
    attention_sink_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute exact self-attention for contiguous rank-3 or rank-4 tensors.

    Supported layouts:
    - `q`: `[seqlen_q, q_heads, head_dim_qk]` or `[batch, seqlen_q, q_heads, head_dim_qk]`
    - `k`: same rank, with `kv_heads` in place of `q_heads`
    - `v`: same rank, with `kv_heads` in place of `q_heads` and `head_dim_vo`

    Returns:
    - `out` with shape `q.shape[:-1] + (head_dim_vo,)` and the same dtype as `q`
    - `lse` with shape `[q_heads, seqlen_q]` or `[batch, q_heads, seqlen_q]`
    """
    if q.ndim not in (3, 4):
        raise ValueError(f"expected rank-3 or rank-4 q tensor, got rank {q.ndim}")
    if q.ndim != k.ndim or q.ndim != v.ndim:
        raise ValueError("q, k, and v must have the same rank")

    squeeze_batch = q.ndim == 3
    if squeeze_batch:
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)

    batch, seqlen_q, q_heads, head_dim_qk = q.shape
    _, seqlen_k, kv_heads, head_dim_k = k.shape
    _, seqlen_v, kv_heads_v, head_dim_v = v.shape
    if head_dim_qk != head_dim_k:
        raise ValueError("q and k must have matching head dims")
    if seqlen_k != seqlen_v or kv_heads != kv_heads_v:
        raise ValueError("k and v must have the same sequence length and head count")
    if q_heads % kv_heads != 0:
        raise ValueError(f"q_heads={q_heads} must be divisible by kv_heads={kv_heads}")

    if softmax_scale is None:
        softmax_scale = head_dim_qk ** -0.5

    q_per_kv = q_heads // kv_heads
    if q_per_kv != 1:
        k = k.repeat_interleave(q_per_kv, dim=2)
        v = v.repeat_interleave(q_per_kv, dim=2)

    q_f = q.permute(0, 2, 1, 3).to(torch.float32)
    k_f = k.permute(0, 2, 1, 3).to(torch.float32)
    v_f = v.permute(0, 2, 1, 3).to(torch.float32)

    scores = torch.matmul(q_f, k_f.transpose(-1, -2)) * float(softmax_scale)
    if causal:
        causal_mask = _causal_mask_right_aligned(seqlen_q, seqlen_k, device=scores.device)
        scores = scores.masked_fill(causal_mask.view(1, 1, seqlen_q, seqlen_k), float("-inf"))
    if window_left >= 0:
        q_idx = torch.arange(seqlen_q, device=scores.device, dtype=torch.int32).view(seqlen_q, 1)
        k_idx = torch.arange(seqlen_k, device=scores.device, dtype=torch.int32).view(1, seqlen_k)
        causal_limit = q_idx + seqlen_k - seqlen_q
        window_mask = k_idx < (causal_limit - int(window_left))
        scores = scores.masked_fill(window_mask.view(1, 1, seqlen_q, seqlen_k), float("-inf"))
    if attention_sink_bias is not None:
        if attention_sink_bias.ndim != 1 or int(attention_sink_bias.shape[0]) != q_heads:
            raise ValueError("attention_sink_bias must have shape [q_heads]")
        sink = attention_sink_bias.to(device=scores.device, dtype=scores.dtype).view(1, q_heads, 1, 1)
        scores_for_lse = torch.cat((scores, sink.expand(batch, q_heads, seqlen_q, 1)), dim=-1)
    else:
        scores_for_lse = scores
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, v_f).permute(0, 2, 1, 3).to(q.dtype)
    if attention_sink_bias is not None:
        denom = torch.exp(scores_for_lse - torch.logsumexp(scores_for_lse, dim=-1, keepdim=True))
        value_probs = denom[..., :seqlen_k]
        out = torch.matmul(value_probs, v_f).permute(0, 2, 1, 3).to(q.dtype)
    lse = torch.logsumexp(scores_for_lse, dim=-1).to(torch.float32)

    if squeeze_batch:
        out = out.squeeze(0)
        lse = lse.squeeze(0)
    return out, lse


def materialize_paged_kv_cache(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    *,
    request_idx: int,
    k_descale: torch.Tensor | None = None,
    v_descale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if k_cache.ndim != 4 or v_cache.ndim != 4:
        raise ValueError("expected paged K/V caches with shape [num_pages, page_size, heads, dim]")
    page_size = int(k_cache.shape[1])
    cache_len = int(cache_seqlens[request_idx].item())
    if cache_len == 0:
        return k_cache[:0].reshape(0, k_cache.shape[2], k_cache.shape[3]), v_cache[:0].reshape(
            0, v_cache.shape[2], v_cache.shape[3]
        )
    num_pages = (cache_len + page_size - 1) // page_size
    page_ids = page_table[request_idx, :num_pages].to(torch.long)
    k = k_cache.index_select(0, page_ids).reshape(num_pages * page_size, k_cache.shape[2], k_cache.shape[3])
    v = v_cache.index_select(0, page_ids).reshape(num_pages * page_size, v_cache.shape[2], v_cache.shape[3])
    k = k[:cache_len]
    v = v[:cache_len]
    if k.dtype == torch.float8_e4m3fn:
        scale = 1.0 if k_descale is None else k_descale[request_idx].view(1, -1, 1)
        k = (k.float() * scale).to(torch.bfloat16)
    if v.dtype == torch.float8_e4m3fn:
        scale = 1.0 if v_descale is None else v_descale[request_idx].view(1, -1, 1)
        v = (v.float() * scale).to(torch.bfloat16)
    return k, v


def paged_attention_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    *,
    k_descale: torch.Tensor | None = None,
    v_descale: torch.Tensor | None = None,
    softmax_scale: float | None = None,
    causal: bool = True,
    window_left: int = -1,
    attention_sink_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference paged self-attention for the SGLang serving contract.

    Inputs:
    - `q`: `[total_q, q_heads, head_dim]`
    - `k_cache`, `v_cache`: `[num_pages, page_size, kv_heads, head_dim]`
    - `page_table`: `[batch, max_pages]`
    - `cache_seqlens`: `[batch]`
    - `cu_seqlens_q`: `[batch + 1]`

    Returns:
    - `out`: `[total_q, q_heads, head_dim]`
    - `lse`: `[total_q, q_heads]` token-major float32
    """
    if q.ndim != 3:
        raise ValueError(f"expected rank-3 q tensor, got rank {q.ndim}")
    if cu_seqlens_q.ndim != 1 or cache_seqlens.ndim != 1:
        raise ValueError("cu_seqlens_q and cache_seqlens must be rank-1 tensors")
    total_q, q_heads, head_dim_qk = q.shape
    head_dim_vo = int(v_cache.shape[-1])
    if softmax_scale is None:
        softmax_scale = head_dim_qk ** -0.5

    out = torch.empty((total_q, q_heads, head_dim_vo), dtype=q.dtype, device=q.device)
    lse = torch.empty((total_q, q_heads), dtype=torch.float32, device=q.device)
    q_offsets = [int(v) for v in cu_seqlens_q.detach().cpu().tolist()]
    for request_idx, (q_start, q_end) in enumerate(zip(q_offsets[:-1], q_offsets[1:])):
        if q_end == q_start:
            continue
        k, v = materialize_paged_kv_cache(
            k_cache,
            v_cache,
            page_table,
            cache_seqlens,
            request_idx=request_idx,
            k_descale=k_descale,
            v_descale=v_descale,
        )
        out_cur, lse_cur = attention_reference(
            q[q_start:q_end],
            k,
            v,
            softmax_scale=softmax_scale,
            causal=causal,
            window_left=window_left,
            attention_sink_bias=attention_sink_bias,
        )
        out[q_start:q_end].copy_(out_cur)
        lse[q_start:q_end].copy_(lse_cur.transpose(0, 1))
    return out, lse
