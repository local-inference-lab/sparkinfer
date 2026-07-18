from __future__ import annotations

import ctypes
import inspect
import warnings

import pytest

import b12x  # noqa: F401 - importing b12x applies the runtime patches under test.
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.base_dsl.compiler import OptLevel
from cutlass.base_dsl.dsl import BaseDSL
from cutlass.base_dsl.jit_executor import ExecutionArgs
from cutlass.base_dsl._mlir_helpers import op as cutlass_op_helpers
from cutlass.base_dsl.runtime import cuda as cutlass_cuda_runtime
from cutlass.cute.nvgpu.warp import mma

import b12x.cute.compiler as cute_compiler
from b12x.cute.compiler import (
    DimKey,
    KernelCompileSpec,
    TensorKey,
    _build_compile_disk_cache_key,
    _compile_disk_cache_payload,
    _structural_cache_key,
    tensor_key,
)
from b12x.cute.runtime_patches import apply_cutlass_runtime_patches
from b12x.cute.utils import make_ptr


def test_compile_only_cache_warning_is_suppressed() -> None:
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        BaseDSL.print_warning(
            object(), "Cache is disabled as user wants to compile only."
        )

    assert captured == []


def test_other_cutlass_warnings_still_emit() -> None:
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        BaseDSL.print_warning(object(), "some other warning")

    assert len(captured) == 1
    assert str(captured[0].message) == "some other warning"


def test_cutlass_source_locations_do_not_scan_for_enclosing_function(
    monkeypatch,
) -> None:
    def fail_findsource(*args, **kwargs):
        raise AssertionError("CUTLASS source locations must not call findsource")

    monkeypatch.setattr(inspect, "findsource", fail_findsource)
    frame = inspect.currentframe()
    assert frame is not None
    frame_info = cutlass_op_helpers.inspect.getframeinfo(frame)

    assert frame_info.filename == __file__
    assert frame_info.function == (
        "test_cutlass_source_locations_do_not_scan_for_enclosing_function"
    )
    assert frame_info.positions.lineno == frame_info.lineno
    assert frame_info.positions.col_offset is not None
    assert frame_info.code_context is not None
    assert "getframeinfo(frame)" in frame_info.code_context[0]


def test_cutlass_memory_debug_helpers_are_stubbed_when_disabled(monkeypatch) -> None:
    if not hasattr(cutlass_cuda_runtime, "_memory_debug_snapshot") or not hasattr(
        cutlass_cuda_runtime, "_memory_debug_log"
    ):
        pytest.skip("CUTLASS runtime has no memory debug snapshot helper")

    monkeypatch.delenv("CUTLASS_DSL_CUDA_MEMORY_DEBUG", raising=False)
    apply_cutlass_runtime_patches()

    assert getattr(cutlass_cuda_runtime, "_b12x_memory_debug_patched", False)
    assert cutlass_cuda_runtime._memory_debug_snapshot() == {
        "free": None,
        "total": None,
        "used": None,
        "torch_allocated": None,
        "torch_reserved": None,
        "external": None,
        "device": None,
    }
    assert cutlass_cuda_runtime._memory_debug_log("test", {}) is None


def test_cutlass_45_provides_sm121a_blockscaled_mma() -> None:
    archs = {str(arch) for arch in mma.MmaSM120BlockScaledOp.admissible_archs}

    assert "sm_121a" in archs
    assert not hasattr(mma.MmaSM120BlockScaledOp, "_b12x_sm121a_patch")


def test_cutlass_45_adapts_cuda_stream_handles() -> None:
    def kernel(stream: cuda.CUstream) -> None:
        pass

    stream = cuda.CUstream(123)
    execution_args = ExecutionArgs(inspect.signature(kernel), kernel.__name__)
    exe_args, adapted_args = execution_args.generate_execution_args((stream,), {})

    assert len(adapted_args) == 1
    assert exe_args == [stream.getPtr()]
    stream_handle = ctypes.cast(exe_args[0], ctypes.POINTER(ctypes.c_void_p)).contents
    assert stream_handle.value == 123


def test_b12x_pointer_cache_key_is_structural() -> None:
    ptr_a = make_ptr(cutlass.Int32, 16, cute.AddressSpace.gmem, assumed_align=16)
    ptr_b = make_ptr(cutlass.Int32, 32, cute.AddressSpace.gmem, assumed_align=16)

    assert ptr_a.__cache_key__ == ptr_b.__cache_key__


def test_compile_disk_cache_key_ignores_pointer_address_and_stream_value() -> None:
    fake = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (4, 8), assumed_align=4)
    ptr_a = make_ptr(cutlass.Int32, 16, cute.AddressSpace.gmem, assumed_align=16)
    ptr_b = make_ptr(cutlass.Int32, 32, cute.AddressSpace.gmem, assumed_align=16)

    compile_callable = cute.compile

    key_a = _build_compile_disk_cache_key(
        compile_callable,
        test_compile_disk_cache_key_ignores_pointer_address_and_stream_value,
        (fake, ptr_a, 0),
        {},
    )
    key_b = _build_compile_disk_cache_key(
        compile_callable,
        test_compile_disk_cache_key_ignores_pointer_address_and_stream_value,
        (fake, ptr_b, 0),
        {},
    )

    assert key_a == key_b


def test_explicit_compile_spec_ignores_full_compile_signature() -> None:
    fake_a = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (4, 8), assumed_align=4)
    fake_b = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (8, 8), assumed_align=4)
    spec = KernelCompileSpec.from_fields(
        "test.explicit",
        1,
        ("shape_bucket", "small"),
    )

    compile_callable = cute.compile

    key_a = _build_compile_disk_cache_key(
        compile_callable,
        test_explicit_compile_spec_ignores_full_compile_signature,
        (fake_a, 1),
        {},
        compile_spec=spec,
    )
    key_b = _build_compile_disk_cache_key(
        compile_callable,
        test_explicit_compile_spec_ignores_full_compile_signature,
        (fake_b, 2),
        {},
        compile_spec=spec,
    )

    assert key_a == key_b


def test_explicit_compile_spec_includes_compile_kwargs() -> None:
    fake = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (4, 8), assumed_align=4)
    spec = KernelCompileSpec.from_fields(
        "test.explicit",
        1,
        ("shape_bucket", "small"),
    )

    compile_callable = cute.compile

    key_a = _build_compile_disk_cache_key(
        compile_callable,
        test_explicit_compile_spec_includes_compile_kwargs,
        (fake,),
        {"options": "a"},
        compile_spec=spec,
    )
    key_b = _build_compile_disk_cache_key(
        compile_callable,
        test_explicit_compile_spec_includes_compile_kwargs,
        (fake,),
        {"options": "b"},
        compile_spec=spec,
    )

    assert key_a != key_b


def test_explicit_compile_spec_changes_cache_key_when_policy_changes() -> None:
    fake = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (4, 8), assumed_align=4)
    compile_callable = cute.compile

    key_a = _build_compile_disk_cache_key(
        compile_callable,
        test_explicit_compile_spec_changes_cache_key_when_policy_changes,
        (fake,),
        {},
        compile_spec=KernelCompileSpec.from_fields(
            "test.explicit",
            1,
            ("shape_bucket", "small"),
        ),
    )
    key_b = _build_compile_disk_cache_key(
        compile_callable,
        test_explicit_compile_spec_changes_cache_key_when_policy_changes,
        (fake,),
        {},
        compile_spec=KernelCompileSpec.from_fields(
            "test.explicit",
            1,
            ("shape_bucket", "large"),
        ),
    )

    assert key_a != key_b


def test_pod_compile_spec_rejects_legacy_tensor_key() -> None:
    tensor_key = TensorKey(
        name="q",
        dtype="torch.uint32",
        rank=3,
        dims=(DimKey.dynamic(), DimKey.exact(64), DimKey.exact(32)),
        stride=(2048, 32, 1),
        device=("cuda", 0),
    )

    with pytest.raises(TypeError, match="legacy compile-key object"):
        KernelCompileSpec.from_facts("test.pod", 1, ("q", tensor_key))


def test_pod_compile_spec_memory_key_does_not_recurse(monkeypatch) -> None:
    spec = KernelCompileSpec.from_facts(
        "test.pod.memory",
        1,
        ("variant", "prefill512"),
        ("q_heads", 64),
    )

    def fail_shape_key(*args, **kwargs):
        raise AssertionError("POD explicit spec should not use recursive shape key")

    monkeypatch.setattr(cute_compiler, "_compile_spec_shape_key", fail_shape_key)

    key = cute_compiler._compile_memory_cache_key(
        cute.compile,
        test_pod_compile_spec_memory_key_does_not_recurse,
        (object(),),
        {},
        spec,
    )

    assert key == ("b12x_cute_memory_cache_v2_explicit_spec", spec.hash_key)


def test_pod_compile_spec_disk_payload_uses_json_key(monkeypatch) -> None:
    spec = KernelCompileSpec.from_facts(
        "test.pod.disk",
        1,
        ("variant", "decode"),
        ("q_heads", 8),
    )

    def fail_shape_key(*args, **kwargs):
        raise AssertionError("POD explicit spec should not use recursive shape key")

    monkeypatch.setattr(cute_compiler, "_compile_spec_shape_key", fail_shape_key)

    payload = _compile_disk_cache_payload(
        cute.compile,
        test_pod_compile_spec_disk_payload_uses_json_key,
        (object(),),
        {},
        compile_spec=spec,
    )

    assert payload[0] == "b12x_cute_compile_cache_v5_explicit_spec"
    assert payload[4] == spec.hash_key
    assert payload[5] == spec.json_key


def test_tensor_key_helper_builds_pod_compile_spec(monkeypatch) -> None:
    q = torch.empty((2, 64), dtype=torch.bfloat16)
    spec = KernelCompileSpec.from_fields(
        "test.tensor_key.pod",
        1,
        tensor_key(
            "q",
            q,
            dims=(DimKey.dynamic(), DimKey.bucket(64)),
            align=16,
        ),
    )

    assert not spec.legacy
    assert spec.fields == ()
    assert '"dynamic"' in spec.json_key
    assert '"bucket"' in spec.json_key

    def fail_shape_key(*args, **kwargs):
        raise AssertionError("POD tensor_key spec should not use recursive shape key")

    monkeypatch.setattr(cute_compiler, "_compile_spec_shape_key", fail_shape_key)
    key = cute_compiler._compile_memory_cache_key(
        cute.compile,
        test_tensor_key_helper_builds_pod_compile_spec,
        (object(),),
        {},
        spec,
    )

    assert key == ("b12x_cute_memory_cache_v2_explicit_spec", spec.hash_key)


def test_from_key_promotes_nested_pod_key_fields() -> None:
    q = torch.empty((2, 64), dtype=torch.bfloat16)
    tensor_fields = (
        tensor_key(
            "q",
            q,
            dims=(DimKey.dynamic(), DimKey.exact(64)),
            align=16,
        ),
    )

    spec = KernelCompileSpec.from_key(
        "test.nested.key_fields",
        1,
        (
            ("policy", "decode"),
            tensor_fields,
        ),
    )

    assert not spec.legacy
    assert '"field"' in spec.json_key
    assert '"q"' in spec.json_key


def test_compile_miss_log_includes_target_attrs_and_arg_shapes(
    capsys, monkeypatch
) -> None:
    monkeypatch.delenv("B12X_LOG_CUTE_COMPILE_STACK", raising=False)
    monkeypatch.setenv("B12X_LOG_CUTE_COMPILES", "1")

    class FakeKernel:
        def __init__(self) -> None:
            self.m = 16
            self.n = 4096
            self.tile = (64, 128)
            self._private = "hidden"

        def __call__(self) -> None:
            pass

    fake = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (4, 8), assumed_align=4)

    cute_compiler._log_cute_compile_miss(
        FakeKernel(),
        (fake, 7),
        {},
        cache_status="disk-cache-miss",
        cache_payload=_compile_disk_cache_payload(
            cute.compile, FakeKernel(), (fake, 7), {}
        ),
    )

    out = capsys.readouterr().out
    assert "[b12x cute.compile] miss" in out
    assert "FakeKernel" in out
    assert "'m': 16" in out
    assert "'n': 4096" in out
    assert "'shape': '(4, 8)'" in out
    assert "'align': 4" in out
    assert "key_inputs=" in out
    assert "'_private': 'hidden'" in out
    assert " cache=" not in out
    assert "python_stack" not in out


def test_explicit_spec_compile_miss_log_is_pod_first(capsys, monkeypatch) -> None:
    monkeypatch.delenv("B12X_LOG_CUTE_COMPILE_ARGS", raising=False)
    monkeypatch.delenv("B12X_LOG_CUTE_COMPILE_STACK", raising=False)
    monkeypatch.setenv("B12X_LOG_CUTE_COMPILES", "1")

    class FakeKernel:
        def __call__(self) -> None:
            pass

    fake = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (4, 8), assumed_align=4)
    spec = KernelCompileSpec.from_facts(
        "test.explicit.log",
        1,
        ("variant", "decode"),
        ("q_heads", 8),
    )

    cute_compiler._log_cute_compile_miss(
        FakeKernel(),
        (fake, 7),
        {},
        cache_status="disk-cache-miss",
        cache_payload=_compile_disk_cache_payload(
            cute.compile, FakeKernel(), (fake, 7), {}, compile_spec=spec
        ),
        cache_key="abcdef0123456789",
    )

    out = capsys.readouterr().out
    assert "[b12x cute.compile] miss" in out
    assert "FakeKernel" in out
    assert f"spec_hash={spec.hash_key}" in out
    assert f"spec={spec.json_key}" in out
    assert "key_inputs=" not in out
    assert "args=" not in out


def test_explicit_spec_compile_miss_log_can_include_args(capsys, monkeypatch) -> None:
    monkeypatch.setenv("B12X_LOG_CUTE_COMPILE_ARGS", "1")
    monkeypatch.delenv("B12X_LOG_CUTE_COMPILE_STACK", raising=False)

    class FakeKernel:
        def __call__(self) -> None:
            pass

    fake = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (4, 8), assumed_align=4)
    spec = KernelCompileSpec.from_facts(
        "test.explicit.log.args",
        1,
        ("variant", "decode"),
    )

    cute_compiler._log_cute_compile_miss(
        FakeKernel(),
        (fake, 7),
        {},
        cache_status="disk-cache-miss",
        cache_payload=_compile_disk_cache_payload(
            cute.compile, FakeKernel(), (fake, 7), {}, compile_spec=spec
        ),
    )

    out = capsys.readouterr().out
    assert "args=" in out
    assert "'shape': '(4, 8)'" in out


def test_compile_miss_log_can_include_python_stack(capsys, monkeypatch) -> None:
    monkeypatch.setenv("B12X_LOG_CUTE_COMPILE_STACK", "1")

    class FakeKernel:
        def __call__(self) -> None:
            pass

    def call_logger() -> None:
        cute_compiler._log_cute_compile_miss(
            FakeKernel(),
            (),
            {},
            cache_status="disk-cache-miss",
            cache_payload=_compile_disk_cache_payload(
                cute.compile, FakeKernel(), (), {}
            ),
        )

    call_logger()

    out = capsys.readouterr().out
    assert "[b12x cute.compile] python_stack" in out
    assert "call_logger" in out
    assert "test_cutlass_runtime_patches.py" in out


def test_run_compiled_reuses_cached_default_executor() -> None:
    executor_calls = []

    class FakeExecutor:
        def run_compiled_program(self, exe_args):
            executor_calls.append(tuple(exe_args))
            return f"executor-call-{len(executor_calls)}"

    class FakeCompiled:
        def __init__(self) -> None:
            self._default_executor = None
            self.to_calls = 0

        def generate_execution_args(self, *args):
            return list(args), []

        def to(self, device):
            assert device is None
            self.to_calls += 1
            return FakeExecutor()

        def run_compiled_program(self, exe_args):
            raise AssertionError("run_compiled should use the cached executor")

    compiled = FakeCompiled()

    assert cute_compiler.run_compiled(compiled, (1, 2)) == "executor-call-1"
    assert cute_compiler.run_compiled(compiled, (3,)) == "executor-call-2"
    assert compiled.to_calls == 1
    assert executor_calls == [(1, 2), (3,)]
    assert compiled._default_executor is not None


def test_run_compiled_falls_back_without_executor_factory() -> None:
    class FakeCompiled:
        def __init__(self) -> None:
            self.calls = []

        def generate_execution_args(self, *args):
            return list(args), []

        def run_compiled_program(self, exe_args):
            self.calls.append(tuple(exe_args))
            return "fallback"

    compiled = FakeCompiled()

    assert cute_compiler.run_compiled(compiled, (1, 2)) == "fallback"
    assert compiled.calls == [(1, 2)]


def test_compile_disk_cache_key_changes_with_compile_env(monkeypatch) -> None:
    fake = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (4, 8), assumed_align=4)
    compile_callable = cute.compile

    cute_compiler.clear_compile_cache()
    monkeypatch.delenv("NVCC_PREPEND_FLAGS", raising=False)
    key_a = _build_compile_disk_cache_key(
        compile_callable,
        test_compile_disk_cache_key_changes_with_compile_env,
        (fake, 0),
        {},
    )

    cute_compiler.clear_compile_cache()
    monkeypatch.setenv("NVCC_PREPEND_FLAGS", "--use_fast_math")
    key_b = _build_compile_disk_cache_key(
        compile_callable,
        test_compile_disk_cache_key_changes_with_compile_env,
        (fake, 0),
        {},
    )

    assert key_a != key_b


def test_compile_disk_cache_key_changes_with_toolchain_key(monkeypatch) -> None:
    fake = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (4, 8), assumed_align=4)
    compile_callable = cute.compile

    cute_compiler.clear_compile_cache()
    monkeypatch.setattr(
        cute_compiler,
        "_runtime_toolchain_key",
        lambda: (("cutlass_dsl", "4.5.0"),),
    )
    key_a = _build_compile_disk_cache_key(
        compile_callable,
        test_compile_disk_cache_key_changes_with_toolchain_key,
        (fake, 0),
        {},
    )

    cute_compiler.clear_compile_cache()
    monkeypatch.setattr(
        cute_compiler,
        "_runtime_toolchain_key",
        lambda: (("cutlass_dsl", "4.5.1"),),
    )
    key_b = _build_compile_disk_cache_key(
        compile_callable,
        test_compile_disk_cache_key_changes_with_toolchain_key,
        (fake, 0),
        {},
    )

    assert key_a != key_b


def test_b12x_package_fingerprint_is_process_static(monkeypatch) -> None:
    calls = 0

    def fake_compute_fingerprint():
        nonlocal calls
        calls += 1
        return f"fingerprint-{calls}"

    cute_compiler._b12x_package_fingerprint.cache_clear()
    cute_compiler._static_compile_cache_context.cache_clear()
    monkeypatch.setattr(
        cute_compiler, "_compute_b12x_package_fingerprint", fake_compute_fingerprint
    )

    assert cute_compiler._b12x_package_fingerprint() == "fingerprint-1"
    assert cute_compiler._b12x_package_fingerprint() == "fingerprint-1"
    assert calls == 1
    cute_compiler._b12x_package_fingerprint.cache_clear()


def test_b12x_compile_uses_memory_cache_when_disk_disabled(monkeypatch) -> None:
    monkeypatch.setenv("B12X_CUTE_COMPILE_DISK_CACHE", "0")
    monkeypatch.delenv("B12X_CUTE_COMPILE_MEMORY_CACHE", raising=False)
    cute_compiler.clear_compile_cache()

    calls = []

    def fake_compile(func, *args, **kwargs):
        calls.append((func, args, kwargs))
        return object()

    class FakeKernel:
        def __call__(self) -> None:
            pass

    monkeypatch.setattr(cute, "compile", fake_compile)
    kernel = FakeKernel()

    compiled_a = cute_compiler.compile(kernel, 1, mode=True)
    compiled_b = cute_compiler.compile(kernel, 1, mode=True)

    assert compiled_a is compiled_b
    assert len(calls) == 1
    info = cute_compiler.compile_cache_info()
    assert info["memory_cache_hits"] == 1
    assert info["compile_misses"] == 1


def test_dsl_compile_options_use_stable_cache_key(monkeypatch) -> None:
    captured_keys = []
    cached = object()

    def capture_memory_key(compile_callable, func, args, kwargs, compile_spec):
        captured_keys.append(kwargs["__dsl_compile_options_key"])
        return ("captured", len(captured_keys))

    monkeypatch.setattr(
        cute_compiler, "_compile_memory_cache_key", capture_memory_key
    )
    monkeypatch.setattr(cute_compiler, "_memory_cache_get", lambda key: cached)

    def kernel() -> None:
        pass

    assert cute_compiler.compile(
        kernel, dsl_compile_options=OptLevel(2)
    ) is cached
    assert cute_compiler.compile(
        kernel, dsl_compile_options=OptLevel(2)
    ) is cached
    assert cute_compiler.compile(
        kernel, dsl_compile_options=OptLevel(3)
    ) is cached

    assert captured_keys[0] == captured_keys[1]
    assert captured_keys[0] != captured_keys[2]
    assert captured_keys[0][:3] == (
        "object",
        "cutlass.base_dsl.compiler",
        "OptLevel",
    )
    assert captured_keys[0][3] == (("_value", 2),)
    assert "object at" not in repr(captured_keys[0])


def test_compile_progress_prints_before_after_and_running_total(
    capsys, monkeypatch
) -> None:
    monkeypatch.setenv("B12X_PRINT_COMPILE_PROGRESS", "1")
    monkeypatch.setenv("B12X_CUTE_COMPILE_DISK_CACHE", "0")
    monkeypatch.setenv("B12X_CUTE_COMPILE_MEMORY_CACHE", "0")
    cute_compiler.clear_compile_cache()

    calls = []

    def fake_compile(func, *args, **kwargs):
        calls.append((func, args, kwargs))
        return object()

    class FakeKernel:
        def __call__(self) -> None:
            pass

    times = iter((10.0, 10.25, 20.0, 20.5))
    monkeypatch.setattr(cute, "compile", fake_compile)
    monkeypatch.setattr(cute_compiler.time, "perf_counter", lambda: next(times))
    spec = KernelCompileSpec.from_facts(
        "test.compile.progress",
        3,
        ("tile", (64, 128)),
        ("stages", 4),
    )

    cute_compiler.compile(FakeKernel(), 1, compile_spec=spec)
    cute_compiler.compile(FakeKernel(), 2, compile_spec=spec)

    lines = capsys.readouterr().out.splitlines()
    assert len(calls) == 2
    assert len(lines) == 4
    assert "compile-start number=1" in lines[0]
    assert "target=" in lines[0]
    assert "kernel=test.compile.progress" in lines[0]
    assert "version=3" in lines[0]
    assert "tile" in lines[0]
    assert "cache_key=" in lines[0]
    assert "compile-done number=1 duration_s=0.250 total_compile_s=0.250" in lines[1]
    assert "compile-start number=2" in lines[2]
    assert "compile-done number=2 duration_s=0.500 total_compile_s=0.750" in lines[3]


def test_compile_progress_prints_after_failed_compile(capsys, monkeypatch) -> None:
    monkeypatch.setenv("B12X_PRINT_COMPILE_PROGRESS", "yes")
    monkeypatch.setenv("B12X_CUTE_COMPILE_DISK_CACHE", "0")
    monkeypatch.setenv("B12X_CUTE_COMPILE_MEMORY_CACHE", "0")
    cute_compiler.clear_compile_cache()

    def fake_compile(*args, **kwargs):
        raise RuntimeError("compiler exploded")

    class FakeKernel:
        def __call__(self) -> None:
            pass

    times = iter((3.0, 4.5))
    monkeypatch.setattr(cute, "compile", fake_compile)
    monkeypatch.setattr(cute_compiler.time, "perf_counter", lambda: next(times))

    with pytest.raises(RuntimeError, match="compiler exploded"):
        cute_compiler.compile(FakeKernel(), 1)

    out = capsys.readouterr().out
    assert "compile-start number=1" in out
    assert "compile-failed number=1 duration_s=1.500 total_compile_s=1.500" in out
    assert "error=RuntimeError: RuntimeError('compiler exploded')" in out


def test_explicit_spec_memory_hit_uses_lightweight_shape_key(monkeypatch) -> None:
    monkeypatch.setenv("B12X_CUTE_COMPILE_DISK_CACHE", "0")
    monkeypatch.delenv("B12X_CUTE_COMPILE_MEMORY_CACHE", raising=False)
    cute_compiler.clear_compile_cache()

    calls = []

    def fake_compile(func, *args, **kwargs):
        compiled = object()
        calls.append(compiled)
        return compiled

    class FakeKernel:
        def __call__(self) -> None:
            pass

    spec = KernelCompileSpec.from_fields(
        "test.explicit.memory",
        1,
        (
            "q",
            TensorKey(
                name="q",
                dtype="torch.bfloat16",
                rank=2,
                dims=(DimKey.exact(1), DimKey.bucket(64)),
                stride=(64, 1),
                device=("cuda", 0),
                align=16,
            ),
        ),
    )

    monkeypatch.setattr(cute, "compile", fake_compile)
    kernel = FakeKernel()

    compiled_a = cute_compiler.compile(kernel, object(), compile_spec=spec)

    def fail_disk_payload(*args, **kwargs):
        raise AssertionError("memory hit should not build disk cache payload")

    def fail_structural_key(*args, **kwargs):
        raise AssertionError("explicit spec hit should not use structural key")

    monkeypatch.setattr(cute_compiler, "_compile_disk_cache_payload", fail_disk_payload)
    monkeypatch.setattr(cute_compiler, "_structural_cache_key", fail_structural_key)

    compiled_b = cute_compiler.compile(kernel, object(), compile_spec=spec)

    assert compiled_a is compiled_b
    assert len(calls) == 1
    assert cute_compiler.compile_cache_info()["memory_cache_hits"] == 1


def test_b12x_compile_can_disable_memory_cache(monkeypatch) -> None:
    monkeypatch.setenv("B12X_CUTE_COMPILE_DISK_CACHE", "0")
    monkeypatch.setenv("B12X_CUTE_COMPILE_MEMORY_CACHE", "0")
    cute_compiler.clear_compile_cache()

    calls = []

    def fake_compile(func, *args, **kwargs):
        compiled = object()
        calls.append(compiled)
        return compiled

    class FakeKernel:
        def __call__(self) -> None:
            pass

    monkeypatch.setattr(cute, "compile", fake_compile)
    kernel = FakeKernel()

    compiled_a = cute_compiler.compile(kernel, 1)
    compiled_b = cute_compiler.compile(kernel, 1)

    assert compiled_a is not compiled_b
    assert len(calls) == 2
    assert cute_compiler.compile_cache_info()["memory_cache_size"] == 0


def test_b12x_compile_disk_hit_populates_memory_cache(monkeypatch) -> None:
    monkeypatch.delenv("B12X_CUTE_COMPILE_DISK_CACHE", raising=False)
    monkeypatch.delenv("B12X_CUTE_COMPILE_MEMORY_CACHE", raising=False)
    cute_compiler.clear_compile_cache()

    compiled = object()
    load_keys = []

    def fake_load(cache_key):
        load_keys.append(cache_key)
        return compiled

    def fail_compile(*args, **kwargs):
        raise AssertionError("disk hit should not call cutlass compile")

    class FakeKernel:
        def __call__(self) -> None:
            pass

    monkeypatch.setattr(cute_compiler, "_load_cute_compile_from_disk", fake_load)
    monkeypatch.setattr(cute, "compile", fail_compile)
    kernel = FakeKernel()

    compiled_a = cute_compiler.compile(kernel, 1)
    compiled_b = cute_compiler.compile(kernel, 1)

    assert compiled_a is compiled
    assert compiled_b is compiled
    assert len(load_keys) == 1
    info = cute_compiler.compile_cache_info()
    assert info["disk_cache_hits"] == 1
    assert info["memory_cache_hits"] == 1


def test_b12x_compile_rechecks_disk_after_cache_key_lock(monkeypatch) -> None:
    monkeypatch.delenv("B12X_CUTE_COMPILE_DISK_CACHE", raising=False)
    monkeypatch.delenv("B12X_CUTE_COMPILE_MEMORY_CACHE", raising=False)
    cute_compiler.clear_compile_cache()

    compiled = object()
    load_keys = []

    def fake_load(cache_key):
        load_keys.append(cache_key)
        return compiled if len(load_keys) == 2 else None

    def fail_compile(*args, **kwargs):
        raise AssertionError("disk recheck hit should not call cutlass compile")

    class FakeKernel:
        def __call__(self) -> None:
            pass

    spec = KernelCompileSpec.from_facts(
        "test.disk.recheck",
        1,
        ("variant", "decode"),
    )

    monkeypatch.setattr(cute_compiler, "_load_cute_compile_from_disk", fake_load)
    monkeypatch.setattr(cute, "compile", fail_compile)

    assert cute_compiler.compile(FakeKernel(), 1, compile_spec=spec) is compiled
    assert len(load_keys) == 2
    info = cute_compiler.compile_cache_info()
    assert info["disk_cache_hits"] == 1
    assert info["compile_misses"] == 0


def test_structural_cache_key_handles_symbolic_fake_compact_tensor_dims() -> None:
    class FakeSymInt:
        def __init__(self, name: str) -> None:
            self.name = name

        def __int__(self) -> int:
            raise TypeError("symbolic dim")

        def __str__(self) -> str:
            return self.name

    FakeCompactTensor = type("_FakeCompactTensor", (), {})
    FakeCompactTensor.__module__ = "cutlass.cute.runtime"
    fake = FakeCompactTensor()
    fake._dtype = cutlass.Int32
    fake._shape = (FakeSymInt("s0"), 8)
    fake._stride_order = (1, 0)
    fake._memspace = cute.AddressSpace.gmem
    fake._assumed_align = 4
    fake._use_32bit_stride = True

    key = _structural_cache_key(fake)

    assert key[0] == "fake_compact_tensor"
    assert key[2][0] == (
        "symbolic_dim",
        FakeSymInt.__module__,
        FakeSymInt.__qualname__,
        "s0",
    )


def test_structural_cache_key_distinguishes_unnamed_cutlass_symbolic_dims() -> None:
    FakeTensor = type("_FakeTensor", (), {})
    FakeTensor.__module__ = "cutlass.cute.runtime"

    fake_a = FakeTensor()
    fake_a._dtype = cutlass.Int32
    fake_a._shape = (cute.sym_int32(divisibility=8), 8)
    fake_a._stride = (8, 1)
    fake_a._memspace = cute.AddressSpace.gmem
    fake_a._assumed_align = 4

    fake_b = FakeTensor()
    fake_b._dtype = cutlass.Int32
    fake_b._shape = (cute.sym_int32(divisibility=8), 8)
    fake_b._stride = (8, 1)
    fake_b._memspace = cute.AddressSpace.gmem
    fake_b._assumed_align = 4

    assert _structural_cache_key(fake_a) != _structural_cache_key(fake_b)


def test_structural_cache_key_skips_warninging_fake_tensor_cache_key() -> None:
    class FakeSymInt:
        def __int__(self) -> int:
            raise TypeError("symbolic dim")

        def __str__(self) -> str:
            return "?{i32 div=8}"

    class FakeTensor:
        __module__ = "some.fake.runtime"

        def __init__(self) -> None:
            self.dtype = cutlass.Int32
            self.shape = (FakeSymInt(), 8)
            self._stride = (8, 1)

        def stride(self):
            return self._stride

        @property
        def __cache_key__(self):
            warnings.warn(
                "FakeTensor cache_key contains unnamed symbolic dimensions. "
                "Different variables with the same shape/stride pattern will have identical cache keys, "
                "which may cause incorrect cache hits.",
                UserWarning,
                stacklevel=2,
            )
            return ("should_not_be_used",)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        key = _structural_cache_key(FakeTensor())

    assert captured == []
    assert key[0] == "fake_tensor"
