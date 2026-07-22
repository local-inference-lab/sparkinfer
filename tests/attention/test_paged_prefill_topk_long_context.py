"""Regression test: paged prefill indexer top-k must stay correct on low-contrast rows.

The tiled radix-select top-k (``SparseNSATiledTopkKernel``, reached via ``index_topk_fp8``
/ ``packed_contiguous``) buckets candidates by the top bits of the score and stores the
threshold bucket in a fixed 4096-slot shared buffer (``_SMEM_CANDS``). When a single
bucket holds more than 4096 candidates -- e.g. many tokens with near-equal scores, as
happens on low-contrast attention rows at longer context -- the pre-fix code silently
dropped the overflow and kept the lowest-indexed (here lowest-scoring) survivors, so it
selected tokens far below the true top-k threshold. The fix re-runs an exact, buffer-free
MSD radix (``_exact_overflow_fallback``) whenever the bucket overflows.

Cases (verified by score, not index, so a miss is genuine and not a tie-break):

* ``seq_len`` sweep: 16384 keeps the hot bucket at 4091 <= 4096 (never overflowed);
  16448 pushes it just over 4096 with almost no extra context, proving the trigger is the
  4096 bucket cap and not the context length; 32768 overflows it hard.
* all-equal scores: one coarse+fine bucket holds every token, so the fallback's tie-fill
  branch must fill the whole top-k from a single pivot key.
* logical output (``output_physical_slots=False``): exercises the two-level fold, whose
  level-1 extent-split pass and level-2 ``run_row_topk`` share the same radix class.
* carry fold (``supertile_k`` < context): multi-chunk streaming fold, so the overflow
  fallback runs in the ``is_first=False`` carry path through the virtual value loader.
"""
from __future__ import annotations

import pytest
import torch

from sparkinfer.attention.nsa_indexer._impl import clear_indexer_caches
from sparkinfer.attention.nsa_indexer.paged import (
    index_topk_fp8,
    pack_paged_index_k_cache_reference,
    prepare_paged_indexer_metadata,
)
from tests.attention.test_paged_indexer_integration import (
    _bind_paged_indexer,
    _make_real_page_table,
    _paged_index_logits,
)

_PAGE = 64
_ROWS = 16
_NUM_HEADS = 32
_TOPK = 2048
_PAGE_START = 3


def _build_scene(device: torch.device, seq_len: int, scores: str) -> dict:
    """Build a paged prefill scene plus its fp32 reference top-k threshold.

    ``scores="monotonic"`` gives each token a strictly increasing score so the true
    top-k is unambiguous; ``scores="equal"`` gives every token the same score, which
    lands the whole row in one radix bucket (degenerate tie-fill).
    """
    width_blocks = (seq_len + _PAGE - 1) // _PAGE
    total_pages = _PAGE_START + width_blocks + 8

    q_fp8 = torch.full(
        (_ROWS, _NUM_HEADS, 128), 0.5, dtype=torch.float32, device=device
    ).to(torch.float8_e4m3fn)
    weights = torch.ones((_ROWS, _NUM_HEADS), dtype=torch.float32, device=device)
    if scores == "monotonic":
        token_scores = torch.linspace(
            0.25, 1.25, seq_len, dtype=torch.float32, device=device
        )
    elif scores == "equal":
        token_scores = torch.full((seq_len,), 0.5, dtype=torch.float32, device=device)
    else:
        raise ValueError(f"unknown scores mode {scores!r}")
    raw_k = torch.zeros((total_pages * _PAGE, 128), dtype=torch.float32, device=device)
    raw_k[_PAGE_START * _PAGE : _PAGE_START * _PAGE + seq_len] = token_scores.unsqueeze(
        1
    ).expand(-1, 128)
    index_k_cache = pack_paged_index_k_cache_reference(raw_k)

    real_page_table = _make_real_page_table(
        page_starts=[_PAGE_START] * _ROWS,
        seqlens=[seq_len] * _ROWS,
        width_blocks=width_blocks,
        device=device,
    )
    shared_page_table = _make_real_page_table(
        page_starts=[_PAGE_START],
        seqlens=[seq_len],
        width_blocks=width_blocks,
        device=device,
    )
    seqlens = torch.full((_ROWS,), seq_len, dtype=torch.int32, device=device)

    ref_logits = _paged_index_logits(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        real_page_table=real_page_table,
        seqlens=seqlens,
    )
    kth_score = torch.topk(ref_logits, _TOPK, dim=1).values[:, -1]
    return {
        "seq_len": seq_len,
        "width_blocks": width_blocks,
        "q_fp8": q_fp8,
        "weights": weights,
        "index_k_cache": index_k_cache,
        "shared_page_table": shared_page_table,
        "seqlens": seqlens,
        "ref_logits": ref_logits,
        "kth_score": kth_score,
    }


def _run_indexer(
    monkeypatch,
    scene: dict,
    *,
    supertile_k: int,
    output_physical_slots: bool,
) -> torch.Tensor:
    monkeypatch.setenv("SPARKINFER_PAGED_INDEX_SUPERTILE_K", str(supertile_k))
    seqlens = scene["seqlens"]
    shared_page_table = scene["shared_page_table"]
    binding = _bind_paged_indexer(
        device=scene["q_fp8"].device,
        num_heads=_NUM_HEADS,
        rows=_ROWS,
        width_blocks=scene["width_blocks"],
        topk=_TOPK,
        real_page_table=shared_page_table.expand(_ROWS, -1),
        seqlens=seqlens,
        supertile_k=supertile_k,
        shared_page_table=True,
        route="packed_contiguous",
        output_physical_slots=output_physical_slots,
    )
    prepare_paged_indexer_metadata(
        real_page_table=shared_page_table.expand(_ROWS, -1),
        cache_seqlens_int32=seqlens,
        expected_num_q_heads=_NUM_HEADS,
        build_schedule=False,
        shared_page_table=True,
    )
    selected = torch.empty((_ROWS, _TOPK), dtype=torch.int32, device=scene["q_fp8"].device)
    clear_indexer_caches()
    index_topk_fp8(
        q_fp8=scene["q_fp8"],
        weights=scene["weights"].unsqueeze(-1),
        index_k_cache=scene["index_k_cache"],
        topk=_TOPK,
        expected_num_q_heads=_NUM_HEADS,
        binding=binding,
        out_indices=selected,
        supertile_k=supertile_k,
    )
    torch.cuda.synchronize(scene["q_fp8"].device)
    return selected


def _assert_selects_true_topk(
    scene: dict, selected: torch.Tensor, *, output_physical_slots: bool
) -> None:
    seq_len = scene["seq_len"]
    if output_physical_slots:
        # Physical slot -> logical (contiguous page table starting at _PAGE_START).
        raw_logical = selected.long() - _PAGE_START * _PAGE
    else:
        # Logical output is already request-relative.
        raw_logical = selected.long()
    # Every selected slot must map to a real in-range token, and a top-k list must not
    # repeat a token. These hold regardless of score, so they -- unlike the score check
    # below -- catch a tie-fill that emits duplicate or out-of-range indices.
    assert bool(((raw_logical >= 0) & (raw_logical < seq_len)).all()), (
        f"selected index outside [0, {seq_len}) at seq_len={seq_len}"
    )
    for r in range(selected.shape[0]):
        row = raw_logical[r].tolist()
        assert len(set(row)) == len(row), (
            f"row {r} selected duplicate indices at seq_len={seq_len}"
        )
    logical = raw_logical.clamp(0, seq_len - 1)
    selected_score = torch.gather(scene["ref_logits"], 1, logical)
    n_below = int(
        (selected_score < (scene["kth_score"][:, None] - 1e-4)).sum().item()
    )
    assert n_below == 0, (
        f"{n_below}/{_ROWS * _TOPK} selected tokens score below the true top-{_TOPK} "
        f"threshold at seq_len={seq_len}"
    )


@pytest.mark.parametrize("seq_len", [16384, 16448, 32768])
def test_paged_prefill_topk_selects_true_topk_at_long_context(
    monkeypatch, seq_len: int
) -> None:
    device = torch.device("cuda")
    scene = _build_scene(device, seq_len, "monotonic")
    selected = _run_indexer(
        monkeypatch, scene, supertile_k=32768, output_physical_slots=True
    )
    _assert_selects_true_topk(scene, selected, output_physical_slots=True)


def test_paged_prefill_topk_all_equal_scores_tie_fill(monkeypatch) -> None:
    """One bucket holds every token: the fallback must tie-fill the whole top-k."""
    device = torch.device("cuda")
    scene = _build_scene(device, 32768, "equal")
    selected = _run_indexer(
        monkeypatch, scene, supertile_k=32768, output_physical_slots=True
    )
    _assert_selects_true_topk(scene, selected, output_physical_slots=True)


def test_paged_prefill_topk_logical_output_two_level_fold(monkeypatch) -> None:
    """output_physical_slots=False routes through the two-level (extent-split + row) fold."""
    device = torch.device("cuda")
    scene = _build_scene(device, 32768, "monotonic")
    selected = _run_indexer(
        monkeypatch, scene, supertile_k=32768, output_physical_slots=False
    )
    _assert_selects_true_topk(scene, selected, output_physical_slots=False)


def test_paged_prefill_topk_carry_fold_overflow(monkeypatch) -> None:
    """supertile_k < context forces the multi-chunk carry fold (is_first=False path).

    All-equal scores guarantee every 8192-token chunk overflows its single bucket, so
    the exact fallback runs through the virtual (carry-aware) value loader.
    """
    device = torch.device("cuda")
    scene = _build_scene(device, 32768, "equal")
    selected = _run_indexer(
        monkeypatch, scene, supertile_k=8192, output_physical_slots=True
    )
    _assert_selects_true_topk(scene, selected, output_physical_slots=True)


def test_paged_prefill_topk_width_does_not_recompile(monkeypatch) -> None:
    """The page-table width must not be part of the tiled top-k compile cache key.

    output_page_table is indexed at runtime as output_page_table[row, page_col]
    (page_col = gidx // output_page_size), so neither shape[1] (the live KV-cache
    page-table width, which grows with sequence/capacity) nor stride(0) (the same
    width for a contiguous page table) is a compile-time input. Pinning either
    into the key re-links a fresh cubin on every capacity bump, per TP rank, on
    the first serving run.

    Two consecutive prefill runs at distinct page-table widths must compile the
    tiled top-k kernel exactly once: the second width reuses the first's cubin.
    """
    from sparkinfer._lib.compiler import clear_compile_cache, compile_cache_info

    device = torch.device("cuda")
    scene_narrow = _build_scene(device, 16384, "monotonic")
    scene_wide = _build_scene(device, 32768, "monotonic")
    width_narrow = scene_narrow["width_blocks"]
    width_wide = scene_wide["width_blocks"]
    assert width_narrow != width_wide, (
        f"test requires distinct page-table widths, got {width_narrow} and {width_wide}"
    )

    # Force every cache key to actually compile (skip the on-disk cache) but keep
    # the in-memory cache on so a repeated identical key is a hit, not a recompile.
    monkeypatch.setenv("SPARKINFER_COMPILE_DISK_CACHE", "0")
    monkeypatch.setenv("SPARKINFER_COMPILE_MEMORY_CACHE", "1")
    clear_compile_cache()

    selected_narrow = _run_indexer(
        monkeypatch, scene_narrow, supertile_k=32768, output_physical_slots=True
    )
    misses_after_narrow = compile_cache_info()["compile_misses"]
    assert misses_after_narrow >= 1, (
        "run A compiled no kernels; the test is not exercising the topk compile path"
    )

    selected_wide = _run_indexer(
        monkeypatch, scene_wide, supertile_k=32768, output_physical_slots=True
    )
    misses_after_wide = compile_cache_info()["compile_misses"]

    assert misses_after_wide == misses_after_narrow, (
        f"page-table width leaked into the compile cache key: width "
        f"{width_narrow} -> {width_wide} triggered "
        f"{misses_after_wide - misses_after_narrow} extra compile(s); "
        f"misses {misses_after_narrow} -> {misses_after_wide}"
    )
    _assert_selects_true_topk(scene_narrow, selected_narrow, output_physical_slots=True)
    _assert_selects_true_topk(scene_wide, selected_wide, output_physical_slots=True)
