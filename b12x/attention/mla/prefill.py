"""Unified SM120 sparse-MLA *prefill* DISPATCHER.

``run_unified_prefill`` is the prefill front door: it infers the model/compute/
scale traits, normalizes inputs, and routes to the FlashInfer-shaped multi-group
(MG) prefill kernel in ``prefill_mg.py`` (``run_unified_prefill_mg``). There is NO
decode-reuse fallback -- an unsupported shape HARD-FAILS (raise-not-fallback,
matching upstream).

Supported (MG) shapes:
  * DSV4/GLM single-cache: heads==16 or heads % 32 == 0
  * DSV4 single-cache: topk in {512, 1024, 2048} (FP8-QK) or 128 (BF16-QK)
    topk==128 also supports odd 16-head multiples by splitting one paired-head
    MG prefix plus one single-group tail.
  * DSV4 dual-cache (extra/indexed tokens): topk==128, heads % 16 == 0,
    pbs_extra in {2, 64} (BF16-QK). Odd 16-head multiples split into a paired
    MG prefix plus one single-group tail.
  * GLM_NSA: topk in {512, 1024, 2048}
Anything else (other topk, unsupported heads, GLM dual, etc.) raises ValueError.
DSV4 + GLM DECODE kernels are untouched and stay byte-identical.
"""

from __future__ import annotations

import os

import torch

from .traits import (
    ComputeMode,
    ModelType,
    ScaleFormat,
    infer_model_type,
    make_unified_traits,
)

# DSV4 compressed contract head dim (q_nope 448 + q_rope 64).
_DSV4_HEAD_DIM = 512
# GLM_NSA uncompressed contract head dim (q_nope 512 + q_rope 64).
_GLM_HEAD_DIM = 576
# GLM per-token packed cache record (reference.pack_mla_kv_cache_reference).
_GLM_KV_GMEM_STRIDE = 656


def run_unified_prefill(
    *,
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    topk_indices: torch.Tensor,
    sm_scale: float,
    page_block_size: int,
    topk_length: torch.Tensor | None = None,
    attn_sink: torch.Tensor | None = None,
    output: torch.Tensor | None = None,
    lse_out: torch.Tensor | None = None,
    stride_kv_block: int | None = None,
    extra_kv_cache: torch.Tensor | None = None,
    extra_indices: torch.Tensor | None = None,
    extra_topk_length: torch.Tensor | None = None,
    extra_page_block_size: int | None = None,
    stride_extra_kv_block: int | None = None,
    workspace=None,
):
    """Unified SM120 sparse-MLA single-pass prefill -> BF16 O + base-2 LSE.

    Pure DISPATCHER: validates the contract, infers traits, and routes to the
    FlashInfer-shaped multi-group (MG) prefill kernel (``run_unified_prefill_mg``).
    Unsupported shapes HARD-FAIL (raise-not-fallback); there is no decode-reuse path.

    Routes DSV4 (q_head_dim==512, UE8M0 footer, V_HAS_ROPE) AND GLM_NSA
    (q_head_dim==576, ARBITRARY_FP32 inline scales, V==nope) through the SAME kernel
    via the traits const_expr branches (model_type/scale_format/v_has_rope/
    nt_per_warp_xv), exactly like the decode launcher. DSV4 additionally supports a
    DUAL-CACHE union (extra_kv_cache / extra_indices / extra_topk_length /
    extra_page_block_size): the CTA attends over the UNION of the MAIN topk cache and
    the EXTRA cache in ONE online softmax (num_main_tiles main chunks then the extra
    chunks). The extra cache is DSV4-only (GLM has no extra section -> RAISE).

    Args:
      q:            (T, heads, D_QK) bf16. D_QK 512 (DSV4) or 576 (GLM_NSA).
      kv_cache:     flat uint8 MAIN KV cache (reshaped to 1-D).
      topk_indices: (T, topk) int32 flat slot ids (-1 = invalid sentinel).
      sm_scale:     softmax scale (typically D_QK**-0.5).
      page_block_size: tokens per MAIN KV block (64 for DSV4/GLM).
      topk_length:  optional (T,) int32 per-token MAIN valid length; entries past it
                    are masked. Defaults to full ``topk`` for every token.
      attn_sink:    optional (heads,) fp32 per-head natural-log sink, folded into
                    the normalizer + base-2 LSE (FlashMLA V4).
      output:       optional pre-allocated (T, heads, D_V) bf16 output (else made).
      lse_out:      optional pre-allocated (T, heads) f32 base-2 LSE (else made).
      stride_kv_block: per-block gmem byte stride for the MAIN cache. Derived from
                    page_block_size + model_type when omitted.
      extra_kv_cache / extra_indices / extra_topk_length / extra_page_block_size:
                    DSV4 dual-cache EXTRA pool (all-or-none; partial trio RAISEs).
      stride_extra_kv_block: EXTRA per-block byte stride (derived when omitted).
      workspace:    unused (prefill is single-pass, no split/merge workspace);
                    accepted for launcher-signature symmetry.

    Returns (O[T, heads, D_V=512] bf16, lse[T, heads] f32 base-2).
    """
    from b12x.attention.mla.compressed_reference import compressed_mla_page_nbytes

    del workspace  # prefill is single-pass; no split/merge workspace needed.

    q_head_dim = int(q.shape[-1])
    if q_head_dim not in (_DSV4_HEAD_DIM, _GLM_HEAD_DIM):
        # Genuinely-unsupported contract -> error like upstream (infer_model_type
        # ICHECKs d_qk in {512, 576}). NOT a legacy fallback.
        raise ValueError(
            f"SM120 sparse MLA prefill supports DSV4 (q_head_dim=512) or GLM_NSA "
            f"(q_head_dim=576); got q_head_dim={q_head_dim}"
        )

    num_tokens, heads, _ = q.shape
    hpb = 16
    if heads % hpb != 0:
        # VALID_HPB<16 small-TP shards are a separate (decode-landed) feature; until
        # ported in prefill this is an unsupported shape -> RAISE (not legacy).
        raise ValueError(
            f"SM120 sparse MLA prefill requires heads divisible by HPB={hpb}, got {heads}"
        )

    model_type, compute_mode, scale_format = infer_model_type(q_head_dim, kv_cache.dtype)
    traits = make_unified_traits(model_type, compute_mode, scale_format)
    d_v = int(traits.d_v)

    # ── DSV4 dual-cache: validate the extra trio (all-or-none) and that it is DSV4. ──
    has_extra = (
        extra_kv_cache is not None
        or extra_indices is not None
        or extra_topk_length is not None
    )
    if has_extra:
        if (
            extra_kv_cache is None
            or extra_indices is None
            or extra_page_block_size is None
        ):
            raise ValueError(
                "SM120 sparse MLA prefill dual-cache requires extra_kv_cache, "
                "extra_indices, and extra_page_block_size together (partial extra "
                "trio is unsupported, matching upstream sparse_mla_sm120.cu:171-174)"
            )
        if model_type != ModelType.DSV4:
            raise ValueError(
                "SM120 sparse MLA prefill dual-cache (extra tokens) is DSV4-only "
                "(q_head_dim==512); GLM/DSV3.2 has no extra cache"
            )

    topk = int(topk_indices.shape[1])

    device = q.device
    if topk_length is None:
        topk_length = torch.full((num_tokens,), topk, dtype=torch.int32, device=device)
    else:
        topk_length = topk_length.to(device=device, dtype=torch.int32).contiguous()

    if stride_kv_block is None:
        if model_type == ModelType.GLM_NSA:
            # GLM cache: per-token 656B contiguous record; a paged "block" holds
            # page_block_size tokens, so the per-block byte stride is pbs*656.
            stride_kv_block = int(page_block_size) * _GLM_KV_GMEM_STRIDE
        else:
            stride_kv_block = int(compressed_mla_page_nbytes(int(page_block_size)))

    q = q.contiguous()
    topk_indices = topk_indices.contiguous()
    if output is None:
        output = torch.empty((num_tokens, heads, d_v), dtype=torch.bfloat16, device=device)
    if lse_out is None:
        lse_out = torch.empty((num_tokens, heads), dtype=torch.float32, device=device)

    # ── MG (multi-head-group) gate ────────────────────────────────────────────
    # DSV4 main-cache. The MG kernel is parameterized by the head-group count
    # ``mg_n_hg`` (a const_expr; one HPB=16 head group per group):
    #   * heads % 32 == 0  -> MG_N_HG=2 (the validated path: one CTA fuses TWO
    #     HPB head groups, sharing the NoPE KV gather across both).
    #   * heads == 16      -> MG_N_HG=1 (small-TP shard; heads % 32 != 0 so it
    #     CANNOT use the 2-group kernel). One CTA owns the single HPB head group:
    #     all group-1 (qk1/acc1/...) work const_expr-elides (clean single-group MG,
    #     16 heads/CTA, replicate_h = heads/16 = 1). This replaces the ~48 GB/s
    #     decode-reuse fallback for the heads==16 shard with an MG-class kernel.
    # Arbitrary single-cache multiples-of-16 that are NOT %32 (48, 80, ...) are
    # not handled by this MG gate. DSV4 dual-cache has a split paired-prefix +
    # single-group-tail path below because there is no decode-reuse fallback.
    #
    # Within each group count, two FlashInfer-shaped QK specializations route:
    #   * topk in {512, 1024, 2048}  -> FP8-QK MG  (block-scaled E4M3 QK).
    #   * topk == 128                -> BF16-QK MG (S0 skips the FP8 Q-quant
    #     prologue; S1 dequants K e4m3->bf16 inline and runs a bf16 m16n8k16 QK;
    #     XV stays FP8). FlashInfer routes topk==128 to this BF16-QK kernel (the
    #     small K-loop where the Q-quant prologue would dominate); it lands a
    #     TIGHTER numeric (no Q-quant loss) than FP8.
    _mg_enabled = os.environ.get(
        "B12X_MLA_SM120_PREFILL_MG", "1"
    ) not in ("0", "false", "False", "off")
    # ── GLM (ARBITRARY_FP32, q=576, v_has_rope=False) MG gate ──────────────────
    # GLM has the SAME FlashInfer MG head-group structure as DSV4 (one CTA fuses
    # MG_N_HG HPB head groups, sharing the KV gather), differing only in the math
    # arms (post-MMA fp32 QK scale, raw-V + 2-pass-W XV, no XV-rope) and the
    # 656/528 KV geometry -- all const_expr-selected in the SAME MG kernel. Route
    # GLM prefill OFF the slow per-head-block decode-reuse path onto MG:
    #   heads == 16        -> MG_N_HG=1 (single-group MG)
    #   heads % 32 == 0    -> MG_N_HG=2 (32/64/128)
    # heads == 8 (MG_N_HG=1 + VALID_HPB=8) is a follow-on (the heads%16!=0 entry
    # guard rejects it before here). topk in {512,1024,2048}.
    _mg_glm = (
        _mg_enabled
        and not has_extra
        and model_type == ModelType.GLM_NSA
        and scale_format == ScaleFormat.ARBITRARY_FP32
    )
    if _mg_glm and topk in (512, 1024, 2048):
        if heads % (2 * hpb) == 0:
            _glm_n_hg = 2
        elif heads == hpb:
            _glm_n_hg = 1
        else:
            _glm_n_hg = 0
        if _glm_n_hg:
            from .prefill_mg import run_unified_prefill_mg

            return run_unified_prefill_mg(
                q=q,
                kv_cache=kv_cache,
                topk_indices=topk_indices,
                sm_scale=sm_scale,
                page_block_size=page_block_size,
                topk_length=topk_length,
                attn_sink=attn_sink,
                output=output,
                lse_out=lse_out,
                stride_kv_block=stride_kv_block,
                compute_mode=ComputeMode.FP8,
                mg_n_hg=_glm_n_hg,
                model_type=ModelType.GLM_NSA,
                scale_format=ScaleFormat.ARBITRARY_FP32,
            )
        if heads > hpb and heads % (2 * hpb) == hpb:
            # GLM TP/DCP layouts can produce odd 16-head multiples after virtual
            # padding, e.g. TP6/DCP3 -> 48 and TP6/DCP6 -> 80. The MG kernel
            # supports paired 32-head launches and single 16-head launches; run
            # both over disjoint output/LSE head ranges instead of rejecting.
            from .prefill_mg import run_unified_prefill_mg

            paired_heads = heads - hpb
            run_unified_prefill_mg(
                q=q,
                kv_cache=kv_cache,
                topk_indices=topk_indices,
                sm_scale=sm_scale,
                page_block_size=page_block_size,
                topk_length=topk_length,
                attn_sink=attn_sink,
                output=output,
                lse_out=lse_out,
                stride_kv_block=stride_kv_block,
                compute_mode=ComputeMode.FP8,
                mg_n_hg=2,
                model_type=ModelType.GLM_NSA,
                scale_format=ScaleFormat.ARBITRARY_FP32,
                active_heads=paired_heads,
                head_offset=0,
            )
            return run_unified_prefill_mg(
                q=q,
                kv_cache=kv_cache,
                topk_indices=topk_indices,
                sm_scale=sm_scale,
                page_block_size=page_block_size,
                topk_length=topk_length,
                attn_sink=attn_sink,
                output=output,
                lse_out=lse_out,
                stride_kv_block=stride_kv_block,
                compute_mode=ComputeMode.FP8,
                mg_n_hg=1,
                model_type=ModelType.GLM_NSA,
                scale_format=ScaleFormat.ARBITRARY_FP32,
                active_heads=hpb,
                head_offset=paired_heads,
            )
    _mg_base = (
        _mg_enabled
        and not has_extra
        and model_type == ModelType.DSV4
        and compute_mode == ComputeMode.FP8  # infer_model_type always FP8 for DSV4
        and scale_format == 0
    )
    if _mg_base and heads % (2 * hpb) == 0:
        _mg_n_hg = 2
    elif _mg_base and heads == hpb:  # heads == 16: single-group MG.
        _mg_n_hg = 1
    else:
        _mg_n_hg = 0  # not MG-eligible for the single-cache gate (incl. 48/80).
    if _mg_n_hg and topk in (512, 1024, 2048):
        from .prefill_mg import run_unified_prefill_mg

        return run_unified_prefill_mg(
            q=q,
            kv_cache=kv_cache,
            topk_indices=topk_indices,
            sm_scale=sm_scale,
            page_block_size=page_block_size,
            topk_length=topk_length,
            attn_sink=attn_sink,
            output=output,
            lse_out=lse_out,
            stride_kv_block=stride_kv_block,
            compute_mode=ComputeMode.FP8,
            mg_n_hg=_mg_n_hg,
        )
    if _mg_n_hg and topk == 128:
        from .prefill_mg import run_unified_prefill_mg

        return run_unified_prefill_mg(
            q=q,
            kv_cache=kv_cache,
            topk_indices=topk_indices,
            sm_scale=sm_scale,
            page_block_size=page_block_size,
            topk_length=topk_length,
            attn_sink=attn_sink,
            output=output,
            lse_out=lse_out,
            stride_kv_block=stride_kv_block,
            compute_mode=ComputeMode.BF16,
            mg_n_hg=_mg_n_hg,
        )
    if _mg_base and topk == 128 and heads > hpb and heads % (2 * hpb) == hpb:
        from .prefill_mg import run_unified_prefill_mg

        paired_heads = heads - hpb
        run_unified_prefill_mg(
            q=q,
            kv_cache=kv_cache,
            topk_indices=topk_indices,
            sm_scale=sm_scale,
            page_block_size=page_block_size,
            topk_length=topk_length,
            attn_sink=attn_sink,
            output=output,
            lse_out=lse_out,
            stride_kv_block=stride_kv_block,
            compute_mode=ComputeMode.BF16,
            mg_n_hg=2,
            active_heads=paired_heads,
            head_offset=0,
        )
        return run_unified_prefill_mg(
            q=q,
            kv_cache=kv_cache,
            topk_indices=topk_indices,
            sm_scale=sm_scale,
            page_block_size=page_block_size,
            topk_length=topk_length,
            attn_sink=attn_sink,
            output=output,
            lse_out=lse_out,
            stride_kv_block=stride_kv_block,
            compute_mode=ComputeMode.BF16,
            mg_n_hg=1,
            active_heads=hpb,
            head_offset=paired_heads,
        )

    # ── DSV4 dual-cache (has_extra) -> MG (BF16-QK), with strip-and-raise. ──────
    # FI ships DSV4 dual-cache as topk==128, BF16-QK. Even 32-head multiples use
    # one paired-head MG launch; odd 16-head multiples (48/80/...) split into a
    # paired prefix plus one single-group tail. Everything else RAISEs (the
    # decode-reuse has_extra body has been removed -- no fallback).
    if has_extra:
        if model_type == ModelType.DSV4 and int(topk) == 128:
            from .prefill_mg import run_unified_prefill_mg

            if heads % (2 * hpb) == 0 or heads == hpb:
                mg_n_hg = 2 if heads % (2 * hpb) == 0 else 1
                return run_unified_prefill_mg(
                    q=q,
                    kv_cache=kv_cache,
                    topk_indices=topk_indices,
                    sm_scale=sm_scale,
                    page_block_size=page_block_size,
                    topk_length=topk_length,
                    attn_sink=attn_sink,
                    output=output,
                    lse_out=lse_out,
                    stride_kv_block=stride_kv_block,
                    compute_mode=ComputeMode.BF16,
                    mg_n_hg=mg_n_hg,
                    extra_kv_cache=extra_kv_cache,
                    extra_indices=extra_indices,
                    extra_topk_length=extra_topk_length,
                    extra_page_block_size=extra_page_block_size,
                    stride_extra_kv_block=stride_extra_kv_block,
                )
            if heads > hpb and heads % (2 * hpb) == hpb:
                paired_heads = heads - hpb
                run_unified_prefill_mg(
                    q=q,
                    kv_cache=kv_cache,
                    topk_indices=topk_indices,
                    sm_scale=sm_scale,
                    page_block_size=page_block_size,
                    topk_length=topk_length,
                    attn_sink=attn_sink,
                    output=output,
                    lse_out=lse_out,
                    stride_kv_block=stride_kv_block,
                    compute_mode=ComputeMode.BF16,
                    mg_n_hg=2,
                    extra_kv_cache=extra_kv_cache,
                    extra_indices=extra_indices,
                    extra_topk_length=extra_topk_length,
                    extra_page_block_size=extra_page_block_size,
                    stride_extra_kv_block=stride_extra_kv_block,
                    active_heads=paired_heads,
                    head_offset=0,
                )
                return run_unified_prefill_mg(
                    q=q,
                    kv_cache=kv_cache,
                    topk_indices=topk_indices,
                    sm_scale=sm_scale,
                    page_block_size=page_block_size,
                    topk_length=topk_length,
                    attn_sink=attn_sink,
                    output=output,
                    lse_out=lse_out,
                    stride_kv_block=stride_kv_block,
                    compute_mode=ComputeMode.BF16,
                    mg_n_hg=1,
                    extra_kv_cache=extra_kv_cache,
                    extra_indices=extra_indices,
                    extra_topk_length=extra_topk_length,
                    extra_page_block_size=extra_page_block_size,
                    stride_extra_kv_block=stride_extra_kv_block,
                    active_heads=hpb,
                    head_offset=paired_heads,
                )
        raise ValueError(
            f"DSV4 dual-cache prefill (heads={heads}, topk={topk}, "
            f"pbs_extra={int(extra_page_block_size)}) requires MG dispatch; only "
            "DSV4 topk==128 with heads divisible by 16 is supported. "
            "No decode-reuse fallback."
        )

    # No MG gate matched. There is NO decode-reuse fallback: an unsupported
    # prefill shape HARD-FAILS (matching upstream's raise-not-fallback contract).
    raise ValueError(
        "SM120 sparse MLA prefill: unsupported shape "
        f"(model_type={int(model_type)}, heads={heads}, topk={topk}, "
        f"compute_mode={int(compute_mode)}, scale_format={int(scale_format)}, "
        f"has_extra={has_extra}, B12X_MLA_SM120_PREFILL_MG={'0' if not _mg_enabled else '1'}). "
        "Supported (MG) shapes: single-cache heads==16 or heads%32==0; "
        "DSV4 single-cache topk in {512, 1024, 2048} (FP8) or 128 "
        "(BF16-QK, heads%16==0); "
        "DSV4 dual-cache topk==128 with heads%16==0 and pbs_extra in {2, 64}; "
        "GLM_NSA topk in {512, 1024, 2048}. No decode-reuse fallback."
    )
