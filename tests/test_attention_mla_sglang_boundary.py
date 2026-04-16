from __future__ import annotations

import importlib
from pathlib import Path
import random
import sys

import pytest
import torch

from b12x.attention.mla.reference import dense_mla_reference, pack_mla_kv_cache_reference

from .helpers import require_sm120
from .test_attention_mla_reference import _compare, _make_glm_case, _require_glm_weights


_SGLANG_PYTHON_ROOT = Path("/home/luke/projects/sglang/python")


def _import_sglang_nsa_backend():
    if not _SGLANG_PYTHON_ROOT.exists():
        pytest.skip(f"sglang sources not found at {_SGLANG_PYTHON_ROOT}")
    root = str(_SGLANG_PYTHON_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        module = importlib.import_module("sglang.srt.layers.attention.nsa_backend")
    except Exception as exc:  # pragma: no cover - environment-dependent import path
        pytest.skip(f"unable to import sglang NSA backend: {exc}")
    return module


def _get_b12x_workspace_method(nsa_backend_module):
    backend_cls = nsa_backend_module.NativeSparseAttnBackend
    method = getattr(backend_cls, "_get_b12x_workspace", None)
    if method is None:
        method = getattr(backend_cls, "_get_b12x_mla_workspace")
    return method


def _get_b12x_forward_method(nsa_backend_module):
    backend_cls = nsa_backend_module.NativeSparseAttnBackend
    method = getattr(backend_cls, "_forward_b12x", None)
    if method is None:
        method = getattr(backend_cls, "_forward_b12x_mla")
    return method


def _get_b12x_decode_kv_reshape_method(nsa_backend_module):
    return nsa_backend_module.NativeSparseAttnBackend._reshape_b12x_decode_kv_rows


def _full_prefix_page_table(*, cache_len: int, rows: int, width: int, device: torch.device) -> torch.Tensor:
    page_table_1 = torch.full((rows, width), -1, dtype=torch.int32, device=device)
    valid = min(cache_len, width)
    if valid > 0:
        page_table_1[:, :valid] = torch.arange(valid, dtype=torch.int32, device=device)
    return page_table_1


def _sample_sparse_page_table(
    *,
    cache_len: int,
    rows: int,
    width: int,
    valid_per_row: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    rng = random.Random(seed)
    page_table_1 = torch.full((rows, width), -1, dtype=torch.int32, device=device)
    population = list(range(cache_len))
    for row_idx in range(rows):
        selected = sorted(rng.sample(population, valid_per_row))
        page_table_1[row_idx, :valid_per_row] = torch.tensor(
            selected, dtype=torch.int32, device=device
        )
    return page_table_1


def _make_fake_backend(
    cfg,
    *,
    device: torch.device,
    topk: int,
    nsa_backend_module,
    num_q_heads: int | None = None,
):
    class _FakeBackend:
        _get_b12x_workspace = _get_b12x_workspace_method(nsa_backend_module)
        _reshape_b12x_decode_kv_rows = _get_b12x_decode_kv_reshape_method(nsa_backend_module)

        def __init__(self):
            self.device = device
            self.q_dtype = torch.bfloat16
            self.kv_cache_dtype = torch.uint8
            self.num_q_heads = cfg.num_heads if num_q_heads is None else int(num_q_heads)
            self.kv_lora_rank = cfg.kv_lora_rank
            self.qk_rope_head_dim = cfg.qk_rope_head_dim
            self.nsa_index_topk = topk
            self.real_page_size = 64
            self.b12x_workspaces: dict[str, object] = {}
            self.b12x_mla_workspaces = self.b12x_workspaces

    return _FakeBackend()


def _forward_b12x_mla(
    nsa_backend_module,
    backend,
    *,
    q_all: torch.Tensor,
    kv_cache: torch.Tensor,
    page_table_1: torch.Tensor,
    metadata: object,
    sm_scale: float,
    v_head_dim: int,
    mode: str,
) -> torch.Tensor:
    forward_b12x = _get_b12x_forward_method(nsa_backend_module)
    return forward_b12x(
        backend,
        q_all=q_all,
        kv_cache=kv_cache,
        page_table_1=page_table_1,
        metadata=metadata,
        sm_scale=sm_scale,
        v_head_dim=v_head_dim,
        mode=mode,
    )


def _make_decode_metadata(*, nsa_backend_module, cache_len: int, page_table_1: torch.Tensor) -> object:
    cache_seqlens = torch.tensor([cache_len], dtype=torch.int32, device=page_table_1.device)
    nsa_cache_seqlens = torch.tensor(
        [int((page_table_1[0] >= 0).sum().item())],
        dtype=torch.int32,
        device=page_table_1.device,
    )
    return nsa_backend_module.NSAMetadata(
        page_size=64,
        cache_seqlens_int32=cache_seqlens,
        max_seq_len_q=1,
        max_seq_len_k=cache_len,
        cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32, device=page_table_1.device),
        cu_seqlens_k=torch.tensor([0, cache_len], dtype=torch.int32, device=page_table_1.device),
        page_table_1=page_table_1,
        real_page_table=page_table_1,
        nsa_cache_seqlens_int32=nsa_cache_seqlens,
        nsa_cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32, device=page_table_1.device),
        nsa_cu_seqlens_k=torch.tensor(
            [0, int(nsa_cache_seqlens[0].item())], dtype=torch.int32, device=page_table_1.device
        ),
        nsa_extend_seq_lens_list=[1],
        nsa_seqlens_expanded=cache_seqlens,
    )


def _make_extend_metadata(
    *,
    nsa_backend_module,
    cache_len: int,
    page_table_1: torch.Tensor,
) -> object:
    rows = page_table_1.shape[0]
    cache_seqlens = torch.full((rows,), cache_len, dtype=torch.int32, device=page_table_1.device)
    valid_per_row = (page_table_1 >= 0).sum(dim=1, dtype=torch.int32)
    nsa_cu = torch.zeros(rows + 1, dtype=torch.int32, device=page_table_1.device)
    nsa_cu[1:] = torch.cumsum(valid_per_row, dim=0)
    return nsa_backend_module.NSAMetadata(
        page_size=64,
        cache_seqlens_int32=cache_seqlens,
        max_seq_len_q=1,
        max_seq_len_k=cache_len,
        cu_seqlens_q=torch.arange(0, rows + 1, dtype=torch.int32, device=page_table_1.device),
        cu_seqlens_k=torch.arange(
            0,
            (rows + 1) * cache_len,
            cache_len,
            dtype=torch.int32,
            device=page_table_1.device,
        ),
        page_table_1=page_table_1,
        real_page_table=page_table_1,
        nsa_cache_seqlens_int32=valid_per_row,
        nsa_cu_seqlens_q=torch.arange(
            0, rows + 1, dtype=torch.int32, device=page_table_1.device
        ),
        nsa_cu_seqlens_k=nsa_cu,
        nsa_extend_seq_lens_list=[1] * rows,
        nsa_seqlens_expanded=cache_seqlens,
    )


def test_sglang_b12x_forward_accepts_page_table_1_alias(monkeypatch) -> None:
    nsa_backend_module = _import_sglang_nsa_backend()
    captured: dict[str, object] = {}

    def fake_decode_forward(**kwargs):
        captured["metadata"] = kwargs["metadata"]
        captured["workspace"] = kwargs["workspace"]
        q_all = kwargs["q_all"]
        return torch.zeros(
            (q_all.shape[0], q_all.shape[1], kwargs["v_head_dim"]),
            dtype=q_all.dtype,
            device=q_all.device,
        )

    monkeypatch.setattr("b12x.integration.mla.sparse_mla_decode_forward", fake_decode_forward)

    class _FakeBackend:
        def _get_b12x_workspace(self, **kwargs):
            captured["workspace_kwargs"] = kwargs
            return object()

    q_all = torch.zeros((1, 8, 256), dtype=torch.bfloat16)
    kv_cache = torch.zeros((4, 1, 656), dtype=torch.uint8)
    page_table_1 = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)
    metadata = _make_decode_metadata(
        nsa_backend_module=nsa_backend_module,
        cache_len=4,
        page_table_1=page_table_1,
    )

    out = nsa_backend_module.NativeSparseAttnBackend._forward_b12x(
        _FakeBackend(),
        q_all=q_all,
        kv_cache=kv_cache,
        page_table_1=page_table_1,
        metadata=metadata,
        sm_scale=0.5,
        v_head_dim=256,
        mode="decode",
    )

    assert out.shape == (1, 8, 256)
    assert captured["workspace"] is not None
    assert captured["workspace_kwargs"] == {
        "mode": "decode",
        "total_q": 1,
        "batch": 1,
        "v_head_dim": 256,
    }
    metadata_arg = captured["metadata"]
    assert torch.equal(metadata_arg.page_table_1, page_table_1)


def test_sglang_b12x_mla_decode_boundary_matches_dense_oracle() -> None:
    device = require_sm120()
    _require_glm_weights()
    nsa_backend_module = _import_sglang_nsa_backend()

    cache_len = 129
    topk = 2048
    cfg, q_all, k_nope, k_rope = _make_glm_case(
        cache_len=cache_len,
        q_len=1,
        seed=71_129,
        device=device,
    )
    packed = pack_mla_kv_cache_reference(k_nope, k_rope)
    page_table_1 = _full_prefix_page_table(cache_len=cache_len, rows=1, width=topk, device=device)
    metadata = _make_decode_metadata(
        nsa_backend_module=nsa_backend_module,
        cache_len=cache_len,
        page_table_1=page_table_1,
    )
    backend = _make_fake_backend(cfg, device=device, topk=topk, nsa_backend_module=nsa_backend_module)

    actual = _forward_b12x_mla(
        nsa_backend_module,
        backend,
        q_all=q_all,
        kv_cache=packed,
        page_table_1=page_table_1,
        metadata=metadata,
        sm_scale=cfg.sm_scale,
        v_head_dim=cfg.kv_lora_rank,
        mode="decode",
    )
    expected = dense_mla_reference(
        q_all=q_all,
        k_nope=k_nope,
        k_rope=k_rope,
        page_table_1=page_table_1,
        sm_scale=cfg.sm_scale,
        v_head_dim=cfg.kv_lora_rank,
    )
    torch.cuda.synchronize(device)

    max_abs, rmse, cos = _compare(actual, expected)
    assert max_abs <= 0.10, f"max_abs={max_abs:.6f}"
    assert rmse <= 0.005, f"rmse={rmse:.6f}"
    assert cos >= 0.9995, f"cos={cos:.6f}"


def test_sglang_b12x_mla_decode_boundary_matches_dense_oracle_for_local_tp_heads() -> None:
    device = require_sm120()
    _require_glm_weights()
    nsa_backend_module = _import_sglang_nsa_backend()

    cache_len = 2050
    topk = 2048
    local_heads = 8
    cfg, q_all, k_nope, k_rope = _make_glm_case(
        cache_len=cache_len,
        q_len=1,
        seed=71_205,
        device=device,
    )
    q_local = q_all[:, :local_heads, :].contiguous()
    packed = pack_mla_kv_cache_reference(k_nope, k_rope)
    page_table_1 = _full_prefix_page_table(cache_len=cache_len, rows=1, width=topk, device=device)
    metadata = _make_decode_metadata(
        nsa_backend_module=nsa_backend_module,
        cache_len=cache_len,
        page_table_1=page_table_1,
    )
    backend = _make_fake_backend(
        cfg,
        device=device,
        topk=topk,
        nsa_backend_module=nsa_backend_module,
        num_q_heads=local_heads,
    )

    actual = _forward_b12x_mla(
        nsa_backend_module,
        backend,
        q_all=q_local,
        kv_cache=packed,
        page_table_1=page_table_1,
        metadata=metadata,
        sm_scale=cfg.sm_scale,
        v_head_dim=cfg.kv_lora_rank,
        mode="decode",
    )
    expected = dense_mla_reference(
        q_all=q_local,
        k_nope=k_nope,
        k_rope=k_rope,
        page_table_1=page_table_1,
        sm_scale=cfg.sm_scale,
        v_head_dim=cfg.kv_lora_rank,
    )
    torch.cuda.synchronize(device)

    max_abs, rmse, cos = _compare(actual, expected)
    assert max_abs <= 0.10, f"max_abs={max_abs:.6f}"
    assert rmse <= 0.005, f"rmse={rmse:.6f}"
    assert cos >= 0.9995, f"cos={cos:.6f}"


def test_sglang_b12x_mla_decode_boundary_matches_dense_oracle_for_local_tp_heads_fp8_view_cache() -> None:
    device = require_sm120()
    _require_glm_weights()
    nsa_backend_module = _import_sglang_nsa_backend()

    cache_len = 2050
    topk = 2048
    local_heads = 8
    cfg, q_all, k_nope, k_rope = _make_glm_case(
        cache_len=cache_len,
        q_len=1,
        seed=71_206,
        device=device,
    )
    q_local = q_all[:, :local_heads, :].contiguous()
    packed = pack_mla_kv_cache_reference(k_nope, k_rope).view(torch.float8_e4m3fn)
    page_table_1 = _full_prefix_page_table(cache_len=cache_len, rows=1, width=topk, device=device)
    metadata = _make_decode_metadata(
        nsa_backend_module=nsa_backend_module,
        cache_len=cache_len,
        page_table_1=page_table_1,
    )
    backend = _make_fake_backend(
        cfg,
        device=device,
        topk=topk,
        nsa_backend_module=nsa_backend_module,
        num_q_heads=local_heads,
    )
    backend.kv_cache_dtype = torch.float8_e4m3fn

    actual = _forward_b12x_mla(
        nsa_backend_module,
        backend,
        q_all=q_local,
        kv_cache=packed,
        page_table_1=page_table_1,
        metadata=metadata,
        sm_scale=cfg.sm_scale,
        v_head_dim=cfg.kv_lora_rank,
        mode="decode",
    )
    expected = dense_mla_reference(
        q_all=q_local,
        k_nope=k_nope,
        k_rope=k_rope,
        page_table_1=page_table_1,
        sm_scale=cfg.sm_scale,
        v_head_dim=cfg.kv_lora_rank,
    )
    torch.cuda.synchronize(device)

    max_abs, rmse, cos = _compare(actual, expected)
    assert max_abs <= 0.10, f"max_abs={max_abs:.6f}"
    assert rmse <= 0.005, f"rmse={rmse:.6f}"
    assert cos >= 0.9995, f"cos={cos:.6f}"


def test_sglang_b12x_mla_extend_boundary_matches_dense_oracle() -> None:
    device = require_sm120()
    _require_glm_weights()
    nsa_backend_module = _import_sglang_nsa_backend()

    cache_len = 2050
    q_len = 4
    topk = 2048
    cfg, q_all, k_nope, k_rope = _make_glm_case(
        cache_len=cache_len,
        q_len=q_len,
        seed=72_050,
        device=device,
    )
    packed = pack_mla_kv_cache_reference(k_nope, k_rope)
    page_table_1 = _sample_sparse_page_table(
        cache_len=cache_len,
        rows=q_len,
        width=topk,
        valid_per_row=topk,
        seed=72050,
        device=device,
    )
    metadata = _make_extend_metadata(
        nsa_backend_module=nsa_backend_module,
        cache_len=cache_len,
        page_table_1=page_table_1,
    )
    backend = _make_fake_backend(cfg, device=device, topk=topk, nsa_backend_module=nsa_backend_module)

    actual = _forward_b12x_mla(
        nsa_backend_module,
        backend,
        q_all=q_all,
        kv_cache=packed,
        page_table_1=page_table_1,
        metadata=metadata,
        sm_scale=cfg.sm_scale,
        v_head_dim=cfg.kv_lora_rank,
        mode="extend",
    )
    expected = dense_mla_reference(
        q_all=q_all,
        k_nope=k_nope,
        k_rope=k_rope,
        page_table_1=page_table_1,
        sm_scale=cfg.sm_scale,
        v_head_dim=cfg.kv_lora_rank,
    )
    torch.cuda.synchronize(device)

    max_abs, rmse, cos = _compare(actual, expected)
    assert max_abs <= 0.10, f"max_abs={max_abs:.6f}"
    assert rmse <= 0.005, f"rmse={rmse:.6f}"
    assert cos >= 0.9995, f"cos={cos:.6f}"
