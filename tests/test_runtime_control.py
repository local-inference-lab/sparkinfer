from __future__ import annotations

from collections import OrderedDict

import pytest

import b12x
import b12x.runtime_control as runtime_control
from b12x.attention.mla.kernel import _run_cached_host_launcher


@pytest.fixture(autouse=True)
def _clear_kernel_resolution_freeze():
    runtime_control.unfreeze_kernel_resolution()
    yield
    runtime_control.unfreeze_kernel_resolution()


def test_b12x_exports_kernel_resolution_freeze_api() -> None:
    assert b12x.freeze_kernel_resolution is runtime_control.freeze_kernel_resolution
    assert b12x.unfreeze_kernel_resolution is runtime_control.unfreeze_kernel_resolution
    assert b12x.freeze_compilation is runtime_control.freeze_kernel_resolution


def test_kernel_resolution_freeze_error_includes_context() -> None:
    runtime_control.freeze_kernel_resolution("warmup complete")

    with pytest.raises(runtime_control.KernelResolutionFrozenError) as excinfo:
        runtime_control.raise_if_kernel_resolution_frozen(
            "cute.compile",
            target=test_kernel_resolution_freeze_error_includes_context,
            cache_key=("shape", 1),
        )

    message = str(excinfo.value)
    assert "cute.compile" in message
    assert "reason=warmup complete" in message
    assert "shape" in message


def test_cached_host_launcher_allows_hits_but_rejects_new_resolution() -> None:
    compile_calls: list[tuple[tuple[object, ...], bool]] = []

    class _Compiled:
        def __init__(self) -> None:
            self.run_count = 0

        def __call__(self, *args):
            exe_args, _ = self.generate_execution_args(*args)
            self.run_compiled_program(exe_args)

        def generate_execution_args(self, *args):
            return args, None

        def run_compiled_program(self, exe_args) -> None:
            assert exe_args == ()
            self.run_count += 1

    def kernel(*args, compile_only=False):
        compile_calls.append((args, compile_only))
        raise AssertionError("kernel resolution should have been blocked")

    compiled = _Compiled()
    kernel._eager_host_launchers = OrderedDict([(("hit",), compiled)])

    runtime_control.freeze_kernel_resolution("warmup complete")

    _run_cached_host_launcher(kernel, ("hit",), ())
    assert compiled.run_count == 1

    with pytest.raises(runtime_control.KernelResolutionFrozenError):
        _run_cached_host_launcher(kernel, ("miss",), ())

    assert compile_calls == []
