from __future__ import annotations

import torch

from b12x.attention.paged.api import _get_cached_plane_tma_descs, _run_cached_host_launcher


class _WorkspaceStub:
    def __init__(self) -> None:
        self._live_plane_tma_desc_cache = {}


def _cache_key(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    plane_cols: int,
    tile_rows: int,
) -> tuple[int, int, tuple[int, ...], tuple[int, ...], int, int]:
    return (
        int(k_cache.data_ptr()),
        int(v_cache.data_ptr()),
        tuple(k_cache.shape),
        tuple(v_cache.shape),
        plane_cols,
        tile_rows,
    )


def test_plane_tma_descriptor_cache_keeps_distinct_layer_bindings() -> None:
    workspace = _WorkspaceStub()
    plane_cols = 64
    tile_rows = 32

    k_cache_1 = torch.empty((4, 64, 2, 128), dtype=torch.bfloat16)
    v_cache_1 = torch.empty((4, 64, 2, 128), dtype=torch.bfloat16)
    k_desc_1 = torch.empty((2, 16), dtype=torch.uint64)
    v_desc_1 = torch.empty((2, 16), dtype=torch.uint64)
    k_ptrs_1 = torch.empty((2,), dtype=torch.int64)
    v_ptrs_1 = torch.empty((2,), dtype=torch.int64)
    workspace._live_plane_tma_desc_cache[_cache_key(
        k_cache_1,
        v_cache_1,
        plane_cols=plane_cols,
        tile_rows=tile_rows,
    )] = (k_desc_1, v_desc_1, k_ptrs_1, v_ptrs_1)

    k_cache_2 = torch.empty((4, 64, 2, 128), dtype=torch.bfloat16)
    v_cache_2 = torch.empty((4, 64, 2, 128), dtype=torch.bfloat16)
    k_desc_2 = torch.empty((2, 16), dtype=torch.uint64)
    v_desc_2 = torch.empty((2, 16), dtype=torch.uint64)
    k_ptrs_2 = torch.empty((2,), dtype=torch.int64)
    v_ptrs_2 = torch.empty((2,), dtype=torch.int64)
    workspace._live_plane_tma_desc_cache[_cache_key(
        k_cache_2,
        v_cache_2,
        plane_cols=plane_cols,
        tile_rows=tile_rows,
    )] = (k_desc_2, v_desc_2, k_ptrs_2, v_ptrs_2)

    assert _get_cached_plane_tma_descs(
        workspace,
        k_cache=k_cache_1,
        v_cache=v_cache_1,
        plane_cols=plane_cols,
        tile_rows=tile_rows,
    ) == (k_desc_1, v_desc_1, k_ptrs_1, v_ptrs_1)
    assert _get_cached_plane_tma_descs(
        workspace,
        k_cache=k_cache_2,
        v_cache=v_cache_2,
        plane_cols=plane_cols,
        tile_rows=tile_rows,
    ) == (k_desc_2, v_desc_2, k_ptrs_2, v_ptrs_2)


def test_cached_host_launcher_warms_compile_cache_during_capture(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    import b12x.attention.paged.api as paged_api

    class _Compiled:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, *args):
            self.calls += 1
            assert args == ("arg",)

    def fake_compile(kernel, *args):
        kernel.compile_calls += 1
        assert args == ("arg",)
        return kernel.compiled

    monkeypatch.setattr(paged_api, "b12x_compile", fake_compile)

    class _Kernel:
        def __init__(self) -> None:
            self.compile_calls = 0
            self.compiled = _Compiled()

        def __call__(self, *args):
            assert args == ("arg",)
            return None

    kernel = _Kernel()
    cache_key = ("shape-only",)

    _run_cached_host_launcher(kernel, cache_key, ("arg",))
    _run_cached_host_launcher(kernel, cache_key, ("arg",))

    assert kernel.compile_calls == 1
    assert kernel.compiled.calls == 2
