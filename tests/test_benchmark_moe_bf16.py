from __future__ import annotations

import inspect
import json

import pytest

from benchmarks import benchmark_moe_bf16
from b12x.integration import tp_moe_bf16


def _write_config(tmp_path, cfg: dict) -> None:
    (tmp_path / "config.json").write_text(json.dumps(cfg))


def test_resolve_activation_uses_profile_defaults() -> None:
    assert benchmark_moe_bf16.resolve_activation(
        benchmark_moe_bf16.MODEL_PROFILES["qwen397b"],
        None,
    ) == "silu"
    assert benchmark_moe_bf16.resolve_activation(
        benchmark_moe_bf16.MODEL_PROFILES["nemotron-backbone"],
        None,
    ) == "relu2"


def test_parser_defaults_to_nemotron_oracle_and_graph_only() -> None:
    parser = benchmark_moe_bf16._build_arg_parser()
    args = parser.parse_args([])

    assert args.model_profile == "nemotron-backbone"
    assert args.validate == "oracle"
    assert "--no-cuda-graph" not in parser._option_string_actions
    assert "--cuda-graph" not in parser._option_string_actions


def test_normalized_result_uses_batch_size_normalized_geomean() -> None:
    result = benchmark_moe_bf16.normalized_result_us(
        {
            1: 1.0,
            4: 4.0,
        }
    )

    assert result == pytest.approx(1000.0)


def test_resolve_activation_rejects_incompatible_override() -> None:
    with pytest.raises(ValueError, match="expects activation='silu'"):
        benchmark_moe_bf16.resolve_activation(
            benchmark_moe_bf16.MODEL_PROFILES["qwen397b"],
            "relu2",
        )
    with pytest.raises(ValueError, match="expects activation='relu2'"):
        benchmark_moe_bf16.resolve_activation(
            benchmark_moe_bf16.MODEL_PROFILES["nemotron-backbone"],
            "silu",
        )


def test_build_model_spec_qwen_uses_checkpoint_shape(tmp_path) -> None:
    _write_config(
        tmp_path,
        {
            "hidden_size": 4096,
            "moe_intermediate_size": 1024,
            "num_experts": 512,
            "num_experts_per_tok": 10,
        },
    )

    spec = benchmark_moe_bf16.build_model_spec(
        tmp_path,
        benchmark_moe_bf16.MODEL_PROFILES["qwen397b"],
    )

    assert spec.hidden_size == 4096
    assert spec.intermediate_size == 1024
    assert spec.num_experts == 512
    assert spec.top_k == 10
    assert spec.tp_size == benchmark_moe_bf16.TP_SIZE


def test_build_model_spec_nemotron_uses_local_hidden_shard(tmp_path) -> None:
    _write_config(
        tmp_path,
        {
            "hidden_size": 16384,
            "moe_intermediate_size": 1024,
            "n_routed_experts": 160,
            "num_experts_per_tok": 8,
        },
    )

    spec = benchmark_moe_bf16.build_model_spec(
        tmp_path,
        benchmark_moe_bf16.MODEL_PROFILES["nemotron-backbone"],
    )

    assert spec.hidden_size == 4096
    assert spec.intermediate_size == 1024
    assert spec.num_experts == 160
    assert spec.top_k == 8
    assert spec.tp_size == 1


def test_resolve_model_path_prefers_bf16_env(monkeypatch, tmp_path) -> None:
    bf16_path = tmp_path / "bf16-env"
    model_path = tmp_path / "model-env"
    bf16_path.mkdir()
    model_path.mkdir()
    monkeypatch.setenv("B12X_BF16_MODEL_PATH", str(bf16_path))
    monkeypatch.setenv("B12X_MODEL_PATH", str(model_path))

    resolved = benchmark_moe_bf16.resolve_model_path(
        benchmark_moe_bf16.MODEL_PROFILES["qwen397b"],
        None,
    )

    assert resolved == bf16_path


def test_bf16_kernel_wrappers_force_fast_math() -> None:
    kernel_cases = [
        (tp_moe_bf16.MoEMicroKernelSilu, (16, (128, 128), 1)),
        (tp_moe_bf16.MoEStaticKernelSilu, (16, (128, 128), 1)),
        (tp_moe_bf16.MoEDynamicKernelSilu, (16, (128, 128))),
        (tp_moe_bf16.MoEMicroKernelRelu2, (16, (128, 128), 1)),
        (tp_moe_bf16.MoEStaticKernelRelu2, (16, (128, 128), 1)),
        (tp_moe_bf16.MoEDynamicKernelRelu2, (16, (128, 128))),
    ]

    for kernel_cls, args in kernel_cases:
        assert "fast_math" not in inspect.signature(kernel_cls.__init__).parameters
        assert kernel_cls(*args).fast_math is True


def test_fail_on_cosine_discrepancy_exits_immediately() -> None:
    metrics = benchmark_moe_bf16.OracleMetrics(
        max_abs=0.0,
        rmse=0.0,
        mean_abs=0.0,
        cos=0.9989,
    )

    with pytest.raises(SystemExit):
        benchmark_moe_bf16.fail_on_cosine_discrepancy("b12x vs oracle", metrics, 1)
