"""Initialization-path tests for ServingEngine runtime policy wiring."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from serve.engine.serving import ServingEngine
from serve.model.attention import B12xPagedAttention


class _FakeTokenizer:
    chat_template = None
    eos_token_id = None
    unk_token_id = -1

    def convert_tokens_to_ids(self, _name):
        return -1


class _FakePagePool:
    def __init__(self, *, num_pages, num_layers, kv_heads, head_dim, kv_dtype, device):
        del num_layers, kv_heads, head_dim, kv_dtype, device
        self.num_pages = num_pages
        self.page_size = 64
        self.k_cache = [torch.zeros(1)]
        self.v_cache = [torch.zeros(1)]

    @staticmethod
    def estimate_num_pages(*_args, **_kwargs):
        return 32


def test_serving_engine_passes_runtime_limits_to_model_runner(monkeypatch):
    cfg = SimpleNamespace(
        num_layers=1,
        num_kv_heads=1,
        head_dim=8,
        vocab_size=16,
        layer_types=None,
    )
    fake_model = SimpleNamespace(config=cfg)
    runner_kwargs = {}

    class _FakeRunner:
        def __init__(self, model, kv_mgr, **kwargs):
            del model, kv_mgr
            runner_kwargs.update(kwargs)

        def warmup(self, *args, **kwargs):
            return None

        def capture_decode_graphs(self, *args, **kwargs):
            return None

        def compile_model(self, *args, **kwargs):
            return None

    monkeypatch.setattr("serve.engine.serving.load_model", lambda *args, **kwargs: fake_model)
    monkeypatch.setattr("serve.engine.serving.AutoTokenizer.from_pretrained", lambda *args, **kwargs: _FakeTokenizer())
    monkeypatch.setattr("serve.engine.serving._estimate_loaded_model_bytes", lambda model: 0)
    monkeypatch.setattr("serve.engine.serving.PagePool", _FakePagePool)
    monkeypatch.setattr("serve.engine.serving.PrefixCheckpointCache", lambda pool, state_arena=None: SimpleNamespace(pool=pool))
    monkeypatch.setattr("serve.engine.serving.start_startup_session", lambda: None)
    monkeypatch.setattr("torch.cuda.mem_get_info", lambda: (8 * 1024**3, 16 * 1024**3))
    monkeypatch.setattr("serve.engine.runner.ModelRunner", _FakeRunner)

    engine = ServingEngine(
        "/tmp/fake-model",
        device="cpu",
        graph_batch_sizes=[],
        prefill_chunk_size=8192,
        max_running=256,
        max_prefill_tokens=16384,
    )

    assert runner_kwargs["max_batch_size"] == 256
    assert runner_kwargs["max_total_tokens"] == 16384
    assert engine.runtime_policy()["max_running"] == 256
    assert engine.runtime_policy()["compile_layers"] is True
    assert engine.runtime_policy()["capture_prefill_graph"] is False


def test_serving_engine_enables_hybrid_layer_compile_without_legacy_warmup(monkeypatch):
    cfg = SimpleNamespace(
        num_layers=2,
        num_kv_heads=1,
        head_dim=8,
        vocab_size=16,
        layer_types=["attention", "linear_attention"],
        linear_num_v_heads=1,
        linear_head_v_dim=8,
        linear_head_k_dim=8,
        linear_num_k_heads=1,
        linear_conv_kernel=4,
    )
    fake_model = SimpleNamespace(config=cfg)
    calls = {"warmup": 0, "compile_model": 0}

    class _FakeRunner:
        def __init__(self, model, kv_mgr, **kwargs):
            del model, kv_mgr, kwargs

        def warmup(self, *args, **kwargs):
            del args, kwargs
            calls["warmup"] += 1

        def capture_decode_graphs(self, *args, **kwargs):
            return None

        def compile_model(self, *args, **kwargs):
            del args, kwargs
            calls["compile_model"] += 1

    class _FakeLinearStateArena:
        def __init__(self, **kwargs):
            del kwargs
            self.num_snapshot_slots = 0

        def zero_all(self):
            return None

        def memory_bytes(self):
            return 0

    monkeypatch.setattr("serve.engine.serving.load_model", lambda *args, **kwargs: fake_model)
    monkeypatch.setattr("serve.engine.serving.AutoTokenizer.from_pretrained", lambda *args, **kwargs: _FakeTokenizer())
    monkeypatch.setattr("serve.engine.serving._estimate_loaded_model_bytes", lambda model: 0)
    monkeypatch.setattr("serve.engine.serving.PagePool", _FakePagePool)
    monkeypatch.setattr("serve.engine.serving.PrefixCheckpointCache", lambda pool, state_arena=None: SimpleNamespace(pool=pool))
    monkeypatch.setattr("serve.engine.serving.start_startup_session", lambda: None)
    monkeypatch.setattr("torch.cuda.mem_get_info", lambda: (8 * 1024**3, 16 * 1024**3))
    monkeypatch.setattr("serve.engine.runner.ModelRunner", _FakeRunner)
    monkeypatch.setattr("serve.cache.linear_state_arena.LinearStateArena", _FakeLinearStateArena)

    engine = ServingEngine(
        "/tmp/fake-model",
        device="cpu",
        graph_batch_sizes=[],
        compile_layers=True,
    )

    assert calls["compile_model"] == 1
    assert calls["warmup"] == 0
    assert engine.runtime_policy()["compile_layers"] is True


def test_serving_engine_skips_decode_warmup_when_decode_graphs_are_enabled(monkeypatch):
    cfg = SimpleNamespace(
        num_layers=1,
        num_kv_heads=1,
        head_dim=8,
        vocab_size=16,
        layer_types=None,
    )
    fake_model = SimpleNamespace(config=cfg)
    warmup_kwargs = []

    class _FakeRunner:
        def __init__(self, model, kv_mgr, **kwargs):
            del model, kv_mgr, kwargs

        def warmup(self, *args, **kwargs):
            del args
            warmup_kwargs.append(dict(kwargs))

        def capture_decode_graphs(self, *args, **kwargs):
            del args, kwargs
            return None

        def compile_model(self, *args, **kwargs):
            del args, kwargs
            return None

    monkeypatch.setattr("serve.engine.serving.load_model", lambda *args, **kwargs: fake_model)
    monkeypatch.setattr("serve.engine.serving.AutoTokenizer.from_pretrained", lambda *args, **kwargs: _FakeTokenizer())
    monkeypatch.setattr("serve.engine.serving._estimate_loaded_model_bytes", lambda model: 0)
    monkeypatch.setattr("serve.engine.serving.PagePool", _FakePagePool)
    monkeypatch.setattr("serve.engine.serving.PrefixCheckpointCache", lambda pool, state_arena=None: SimpleNamespace(pool=pool))
    monkeypatch.setattr("serve.engine.serving.start_startup_session", lambda: None)
    monkeypatch.setattr("torch.cuda.mem_get_info", lambda: (8 * 1024**3, 16 * 1024**3))
    monkeypatch.setattr("serve.engine.runner.ModelRunner", _FakeRunner)

    ServingEngine(
        "/tmp/fake-model",
        device="cpu",
        graph_batch_sizes=[1, 2, 4],
        compile_layers=False,
    )

    assert len(warmup_kwargs) == 1
    assert warmup_kwargs[0]["warm_decode"] is False


def test_serving_engine_does_not_capture_prefill_graph_by_default(monkeypatch):
    cfg = SimpleNamespace(
        num_layers=1,
        num_kv_heads=1,
        head_dim=8,
        vocab_size=16,
        layer_types=None,
    )
    fake_model = SimpleNamespace(config=cfg)
    capture_kwargs = []

    class _FakeRunner:
        def __init__(self, model, kv_mgr, **kwargs):
            del model, kv_mgr, kwargs

        def warmup(self, *args, **kwargs):
            del args, kwargs
            return None

        def capture_decode_graphs(self, *args, **kwargs):
            del args
            capture_kwargs.append(dict(kwargs))

        def compile_model(self, *args, **kwargs):
            del args, kwargs
            return None

    monkeypatch.setattr("serve.engine.serving.load_model", lambda *args, **kwargs: fake_model)
    monkeypatch.setattr("serve.engine.serving.AutoTokenizer.from_pretrained", lambda *args, **kwargs: _FakeTokenizer())
    monkeypatch.setattr("serve.engine.serving._estimate_loaded_model_bytes", lambda model: 0)
    monkeypatch.setattr("serve.engine.serving.PagePool", _FakePagePool)
    monkeypatch.setattr("serve.engine.serving.PrefixCheckpointCache", lambda pool, state_arena=None: SimpleNamespace(pool=pool))
    monkeypatch.setattr("serve.engine.serving.start_startup_session", lambda: None)
    monkeypatch.setattr("torch.cuda.mem_get_info", lambda: (8 * 1024**3, 16 * 1024**3))
    monkeypatch.setattr("serve.engine.runner.ModelRunner", _FakeRunner)

    ServingEngine(
        "/tmp/fake-model",
        device="cpu",
        graph_batch_sizes=[1, 2, 4],
    )

    assert len(capture_kwargs) == 1
    assert capture_kwargs[0]["batch_sizes"] == [1, 2, 4]
    assert capture_kwargs[0]["prefill_chunk_size"] is None


def test_serving_engine_can_opt_into_prefill_graph_capture(monkeypatch):
    cfg = SimpleNamespace(
        num_layers=1,
        num_kv_heads=1,
        head_dim=8,
        vocab_size=16,
        layer_types=None,
    )
    fake_model = SimpleNamespace(config=cfg)
    capture_kwargs = []

    class _FakeRunner:
        def __init__(self, model, kv_mgr, **kwargs):
            del model, kv_mgr, kwargs

        def warmup(self, *args, **kwargs):
            del args, kwargs
            return None

        def capture_decode_graphs(self, *args, **kwargs):
            del args
            capture_kwargs.append(dict(kwargs))

        def compile_model(self, *args, **kwargs):
            del args, kwargs
            return None

    monkeypatch.setattr("serve.engine.serving.load_model", lambda *args, **kwargs: fake_model)
    monkeypatch.setattr("serve.engine.serving.AutoTokenizer.from_pretrained", lambda *args, **kwargs: _FakeTokenizer())
    monkeypatch.setattr("serve.engine.serving._estimate_loaded_model_bytes", lambda model: 0)
    monkeypatch.setattr("serve.engine.serving.PagePool", _FakePagePool)
    monkeypatch.setattr("serve.engine.serving.PrefixCheckpointCache", lambda pool, state_arena=None: SimpleNamespace(pool=pool))
    monkeypatch.setattr("serve.engine.serving.start_startup_session", lambda: None)
    monkeypatch.setattr("torch.cuda.mem_get_info", lambda: (8 * 1024**3, 16 * 1024**3))
    monkeypatch.setattr("serve.engine.runner.ModelRunner", _FakeRunner)

    ServingEngine(
        "/tmp/fake-model",
        device="cpu",
        graph_batch_sizes=[1, 2, 4],
        capture_prefill_graph=True,
        prefill_chunk_size=1024,
    )

    assert len(capture_kwargs) == 1
    assert capture_kwargs[0]["batch_sizes"] == [1, 2, 4]
    assert capture_kwargs[0]["prefill_chunk_size"] == 1024


def test_paged_attention_uses_extend_workspace_for_unsupported_decode_shapes():
    attn = B12xPagedAttention(
        num_q_heads=16,
        num_kv_heads=8,
        head_dim=128,
        hidden_size=2048,
        rotary_dim=128,
        rms_norm_eps=1e-5,
        qkv_weight=torch.zeros(4096, 2048, dtype=torch.bfloat16),
        o_proj_weight=torch.zeros(2048, 2048, dtype=torch.bfloat16),
        q_norm_weight=torch.zeros(128, dtype=torch.bfloat16),
        k_norm_weight=torch.zeros(128, dtype=torch.bfloat16),
    )

    workspaces = attn.allocate_workspaces(
        device="cpu",
        kv_dtype=torch.bfloat16,
        page_size=64,
        num_cache_pages=128,
        max_total_q=4,
        max_batch=4,
        max_page_table_width=128,
        use_cuda_graph=True,
    )

    assert workspaces["decode"] is workspaces["extend"]
    assert workspaces["decode"].mode == "extend"
