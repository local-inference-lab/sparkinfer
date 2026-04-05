"""Tests for serve CLI parsing helpers."""

import sys

from serve.cli import _parse_graph_batch_sizes, main


def test_parse_graph_batch_sizes_defaults_when_enabled():
    assert _parse_graph_batch_sizes(None, enabled=True) == [1, 2, 4, 8]


def test_parse_graph_batch_sizes_dedupes_and_sorts():
    assert _parse_graph_batch_sizes("8,2,2,4", enabled=True) == [2, 4, 8]


def test_parse_graph_batch_sizes_disabled_returns_empty():
    assert _parse_graph_batch_sizes("1,2,4", enabled=False) == []


def test_cli_defaults_to_eager_prefill(monkeypatch):
    launch = {}

    def _fake_launch_tp(fn, *, world_size, args, gpu_ids):
        launch["fn"] = fn
        launch["world_size"] = world_size
        launch["args"] = args
        launch["gpu_ids"] = gpu_ids

    monkeypatch.setattr("serve.cli.launch_tp", _fake_launch_tp)
    monkeypatch.setattr(sys, "argv", ["serve.cli", "/tmp/fake-model"])

    main()

    assert launch["world_size"] == 1
    assert launch["args"][10] is False


def test_cli_can_opt_into_prefill_graph_capture(monkeypatch):
    launch = {}

    def _fake_launch_tp(fn, *, world_size, args, gpu_ids):
        launch["fn"] = fn
        launch["world_size"] = world_size
        launch["args"] = args
        launch["gpu_ids"] = gpu_ids

    monkeypatch.setattr("serve.cli.launch_tp", _fake_launch_tp)
    monkeypatch.setattr(
        sys,
        "argv",
        ["serve.cli", "/tmp/fake-model", "--capture-prefill-graph"],
    )

    main()

    assert launch["world_size"] == 1
    assert launch["args"][10] is True
