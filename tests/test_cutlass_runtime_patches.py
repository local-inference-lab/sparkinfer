from __future__ import annotations

import ctypes
import inspect
import warnings

import b12x  # noqa: F401 - importing b12x applies the runtime patches under test.
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.base_dsl.dsl import BaseDSL
from cutlass.base_dsl.jit_executor import ExecutionArgs
from cutlass.cute.nvgpu.warp import mma

import b12x.cute.compiler as cute_compiler
from b12x.cute.compiler import (
    KernelCompileSpec,
    _build_compile_disk_cache_key,
    _compile_disk_cache_payload,
    _structural_cache_key,
)
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


def test_compile_disk_cache_key_changes_with_compile_env(monkeypatch) -> None:
    fake = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (4, 8), assumed_align=4)
    compile_callable = cute.compile

    monkeypatch.delenv("NVCC_PREPEND_FLAGS", raising=False)
    key_a = _build_compile_disk_cache_key(
        compile_callable,
        test_compile_disk_cache_key_changes_with_compile_env,
        (fake, 0),
        {},
    )

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
