from __future__ import annotations

import pytest
import cutlass.cute as cute

import b12x
import b12x.cute.compiler as cute_compiler
import b12x.runtime_control as runtime_control
from b12x.cute.compiler import KernelCompileSpec


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


def test_cute_launch_allows_cached_hits_but_rejects_new_resolution(monkeypatch) -> None:
    monkeypatch.setenv("B12X_CUTE_COMPILE_DISK_CACHE", "0")
    monkeypatch.delenv("B12X_CUTE_COMPILE_MEMORY_CACHE", raising=False)
    cute_compiler.clear_compile_cache()
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

    def fake_compile(func, *args, compile_only=False, **kwargs):
        del func, kwargs
        compile_calls.append((args, compile_only))
        return compiled

    compiled = _Compiled()
    monkeypatch.setattr(cute, "compile", fake_compile)

    def kernel() -> None:
        raise AssertionError("kernel should run through compiled object")

    hit_spec = KernelCompileSpec.from_fields("test.runtime_control", 1, ("shape", "hit"))
    miss_spec = KernelCompileSpec.from_fields("test.runtime_control", 1, ("shape", "miss"))

    cute_compiler.launch(
        kernel,
        compile_spec=hit_spec,
        compile_args=(),
        runtime_args=(),
    )
    assert compile_calls == [((), False)]
    assert compiled.run_count == 1

    runtime_control.freeze_kernel_resolution("warmup complete")

    cute_compiler.launch(
        kernel,
        compile_spec=hit_spec,
        compile_args=(),
        runtime_args=(),
    )
    assert compiled.run_count == 2

    with pytest.raises(runtime_control.KernelResolutionFrozenError):
        cute_compiler.launch(
            kernel,
            compile_spec=miss_spec,
            compile_args=(),
            runtime_args=(),
        )

    assert compile_calls == [((), False)]
