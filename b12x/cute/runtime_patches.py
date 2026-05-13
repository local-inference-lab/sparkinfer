from __future__ import annotations

import ctypes
import hashlib
import importlib.metadata
import inspect
import os
import sys
from contextlib import suppress
from functools import lru_cache
from functools import wraps
from pathlib import Path
from types import SimpleNamespace
from typing import Any

_COMPILE_ONLY_CACHE_WARNING = "Cache is disabled as user wants to compile only."
_PATCHED = False
_B12X_PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def _cute_compile_disk_cache_enabled() -> bool:
    raw = os.environ.get("B12X_CUTE_COMPILE_DISK_CACHE", "1")
    return raw.lower() not in {"0", "false", "no", ""}


def _cute_compile_cache_dir() -> Path:
    root = os.environ.get("B12X_CUTE_COMPILE_CACHE_DIR")
    if root:
        return Path(root)
    cute_cache_dir = os.environ.get("CUTE_DSL_CACHE_DIR")
    if cute_cache_dir:
        return Path(cute_cache_dir) / "b12x_object_cache"
    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache_home:
        return Path(xdg_cache_home) / "b12x" / "cute_compile"
    return Path.home() / ".cache" / "b12x" / "cute_compile"


def _iter_fingerprint_files(root: Path) -> list[Path]:
    files = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if "__pycache__" in path.parts:
            continue
        if path.suffix in {".pyc", ".pyo"}:
            continue
        files.append(path)
    files.sort()
    return files


def _tree_state(root: Path) -> tuple[tuple[str, int, int], ...]:
    entries = []
    for path in _iter_fingerprint_files(root):
        stat = path.stat()
        entries.append((str(path.relative_to(root)), stat.st_mtime_ns, stat.st_size))
    return tuple(entries)


@lru_cache(maxsize=8)
def _tree_fingerprint_cached(
    root_str: str, state: tuple[tuple[str, int, int], ...]
) -> str:
    root = Path(root_str)
    digest = hashlib.sha256()
    for rel_path, _mtime_ns, _size in state:
        path = root / rel_path
        digest.update(rel_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _tree_fingerprint(root: Path) -> str:
    return _tree_fingerprint_cached(str(root), _tree_state(root))


def _b12x_package_fingerprint() -> str:
    return _tree_fingerprint(_B12X_PACKAGE_ROOT)


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return ""


@lru_cache(maxsize=1)
def _runtime_toolchain_key() -> tuple[object, ...]:
    torch_version = _distribution_version("torch")
    torch_cuda_version = ""
    try:
        import torch

        if not torch_version:
            torch_version = getattr(torch, "__version__", "")
        torch_cuda_version = getattr(torch.version, "cuda", "") or ""
    except Exception:
        pass

    cutlass_version = _distribution_version("nvidia-cutlass-dsl")
    if not cutlass_version:
        cutlass_version = _distribution_version("cutlass")
    if not cutlass_version:
        try:
            import cutlass

            cutlass_version = getattr(cutlass, "__version__", "")
        except Exception:
            cutlass_version = ""

    return (
        ("python", sys.implementation.name, sys.version_info[:3]),
        ("torch", torch_version),
        ("torch_cuda", torch_cuda_version),
        ("cutlass_dsl", cutlass_version),
        (
            "cutlass_dsl_libs_base",
            _distribution_version("nvidia-cutlass-dsl-libs-base"),
        ),
        (
            "cutlass_dsl_libs_cu13",
            _distribution_version("nvidia-cutlass-dsl-libs-cu13"),
        ),
        ("cuda_python", _distribution_version("cuda-python")),
        ("cuda_bindings", _distribution_version("cuda-bindings")),
    )


def _compile_environment_key() -> tuple[tuple[str, str], ...]:
    compile_env_vars = (
        "CC",
        "CXX",
        "CUDA_HOME",
        "CUDA_PATH",
        "CUDA_TOOLKIT_PATH",
        "CUDACXX",
        "CUTE_DSL_ARCH",
        "NVCC_APPEND_FLAGS",
        "NVCC_PREPEND_FLAGS",
    )
    return tuple((name, os.environ.get(name, "")) for name in compile_env_vars)


def _function_fingerprint(func: Any) -> tuple[str, str, str]:
    func = inspect.unwrap(func)
    module = getattr(func, "__module__", "")
    qualname = getattr(
        func, "__qualname__", getattr(func, "__name__", type(func).__qualname__)
    )
    if module == "b12x" or module.startswith("b12x."):
        return module, qualname, f"b12x:{_b12x_package_fingerprint()}"
    try:
        source = inspect.getsource(func)
        payload = source.encode("utf-8")
    except (OSError, TypeError):
        code = getattr(func, "__code__", None)
        if code is None:
            payload = repr(func).encode("utf-8")
        else:
            payload = repr(
                (
                    code.co_code,
                    code.co_consts,
                    code.co_names,
                    code.co_varnames,
                    code.co_argcount,
                    code.co_kwonlyargcount,
                )
            ).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return module, qualname, digest


def _normalize_compile_target(func: Any, visited: set[int]) -> Any:
    if inspect.ismethod(func):
        return (
            "method",
            _function_fingerprint(func.__func__),
            _structural_cache_key(func.__self__, visited),
        )
    if inspect.isfunction(func):
        return ("function", _function_fingerprint(func))
    if callable(func) and hasattr(func.__call__, "__func__"):
        state = vars(func) if hasattr(func, "__dict__") else None
        return (
            "callable_instance",
            type(func).__module__,
            type(func).__qualname__,
            _function_fingerprint(func.__call__.__func__),
            _structural_cache_key(state, visited),
        )
    return ("callable", type(func).__module__, type(func).__qualname__, repr(func))


def _structural_dim_key(dim: Any, visited: set[int]) -> Any:
    if dim is None or isinstance(dim, (bool, int, float, str)):
        return dim
    try:
        return int(dim)
    except (TypeError, ValueError):
        pass
    label = None
    for attr in ("symbol", "_symbol", "name", "_name"):
        value = getattr(dim, attr, None)
        if isinstance(value, str) and value:
            label = value
            break
    if label is None:
        node = getattr(dim, "node", None)
        expr = getattr(node, "expr", getattr(node, "_expr", None))
        if expr is not None:
            label = str(expr)
    if label is None:
        text = str(dim)
        if text and not text.startswith("?{"):
            label = text
    if label is not None:
        return (
            "symbolic_dim",
            type(dim).__module__,
            type(dim).__qualname__,
            label,
        )
    return (
        "symbolic_dim",
        type(dim).__module__,
        type(dim).__qualname__,
        id(getattr(dim, "node", dim)),
    )


def _maybe_call_zero_arg(value: Any) -> Any:
    if callable(value):
        try:
            return value()
        except TypeError:
            return None
    return value


def _first_present_attr(value: Any, *names: str) -> Any:
    for name in names:
        try:
            attr = getattr(value, name)
        except Exception:
            continue
        attr = _maybe_call_zero_arg(attr)
        if attr is not None:
            return attr
    return None


def _tensor_like_cache_key(value: Any, visited: set[int]) -> Any | None:
    shape = _first_present_attr(value, "_shape", "shape")
    if shape is None:
        return None
    stride = _first_present_attr(value, "_stride", "stride")
    stride_order = _first_present_attr(value, "_stride_order", "stride_order")
    dtype = _first_present_attr(value, "_dtype", "dtype", "element_type")
    device = _first_present_attr(value, "fake_device", "device")
    layout = _first_present_attr(value, "layout")
    memspace = _first_present_attr(value, "memspace", "_memspace")
    assumed_align = _first_present_attr(value, "_assumed_align")
    use_32bit_stride = _first_present_attr(value, "_use_32bit_stride")
    shape_key = tuple(_structural_dim_key(dim, visited) for dim in shape)
    stride_key = (
        None
        if stride is None
        else tuple(_structural_dim_key(dim, visited) for dim in stride)
    )
    stride_order_key = (
        None
        if stride_order is None
        else tuple(_structural_dim_key(dim, visited) for dim in stride_order)
    )
    return (
        "fake_tensor",
        type(value).__module__,
        type(value).__qualname__,
        _structural_cache_key(dtype, visited),
        shape_key,
        stride_key,
        stride_order_key,
        _structural_cache_key(device, visited),
        _structural_cache_key(layout, visited),
        _structural_cache_key(memspace, visited),
        assumed_align,
        use_32bit_stride,
    )


def _structural_cache_key(value: Any, visited: set[int] | None = None) -> Any:
    if visited is None:
        visited = set()

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return ("bytes", value.hex())
    if isinstance(value, Path):
        return ("path", str(value))
    if inspect.isfunction(value) or inspect.ismethod(value):
        return _normalize_compile_target(value, visited)
    if isinstance(value, type):
        return ("type", value.__module__, value.__qualname__)
    if isinstance(value, SimpleNamespace):
        return (
            "namespace",
            tuple(
                sorted(
                    (k, _structural_cache_key(v, visited))
                    for k, v in vars(value).items()
                )
            ),
        )
    if isinstance(value, dict):
        return tuple(
            sorted(
                (_structural_cache_key(k, visited), _structural_cache_key(v, visited))
                for k, v in value.items()
            )
        )
    if isinstance(value, (tuple, list)):
        return tuple(_structural_cache_key(v, visited) for v in value)
    if isinstance(value, set):
        return tuple(sorted(_structural_cache_key(v, visited) for v in value))

    type_name = type(value).__name__
    type_module = type(value).__module__
    if type_name == "CUstream" and type_module.startswith("cuda.bindings"):
        return ("cuda_stream",)
    if type_module == "cutlass.cute.runtime" and type_name == "_Tensor":
        dtype = getattr(value, "_dtype", getattr(value, "element_type", None))
        shape = tuple(_structural_dim_key(dim, visited) for dim in value.shape)
        stride = tuple(_structural_dim_key(dim, visited) for dim in value.stride)
        memspace = getattr(value, "memspace", getattr(value, "_memspace", None))
        assumed_align = getattr(value, "_assumed_align", None)
        is_dynamic = getattr(value, "_is_dynamic", None)
        use_32bit_stride = getattr(value, "_use_32bit_stride", None)
        return (
            "runtime_tensor",
            dtype,
            shape,
            stride,
            memspace,
            assumed_align,
            is_dynamic,
            use_32bit_stride,
        )
    if type_module == "cutlass.cute.runtime" and type_name == "_FakeCompactTensor":
        dtype = getattr(value, "_dtype", None)
        shape = tuple(
            _structural_dim_key(dim, visited) for dim in getattr(value, "_shape", ())
        )
        stride_order = tuple(
            _structural_dim_key(dim, visited)
            for dim in getattr(value, "_stride_order", ())
        )
        memspace = getattr(value, "_memspace", None)
        assumed_align = getattr(value, "_assumed_align", None)
        use_32bit_stride = getattr(value, "_use_32bit_stride", None)
        return (
            "fake_compact_tensor",
            dtype,
            shape,
            stride_order,
            memspace,
            assumed_align,
            use_32bit_stride,
        )
    if "FakeTensor" in type_name:
        fake_tensor_key = _tensor_like_cache_key(value, visited)
        if fake_tensor_key is not None:
            return fake_tensor_key

    cache_key_attr = getattr(value, "__cache_key__", None)
    if cache_key_attr is not None:
        return (
            "cache_key",
            type_module,
            type_name,
            _structural_cache_key(cache_key_attr, visited),
        )

    object_id = id(value)
    if object_id in visited:
        return ("cycle", type_module, type_name)

    if hasattr(value, "__dict__"):
        visited.add(object_id)
        try:
            return (
                "object",
                type_module,
                type_name,
                tuple(
                    sorted(
                        (
                            k,
                            _structural_cache_key(v, visited),
                        )
                        for k, v in vars(value).items()
                    )
                ),
            )
        finally:
            visited.remove(object_id)

    return ("repr", type_module, type_name, repr(value))


def _compile_options_cache_key(compile_callable: Any) -> tuple[str, ...]:
    compile_options = getattr(compile_callable, "_compile_options", None)
    if compile_options is None:
        return ()
    options = getattr(compile_options, "options", {})
    serialized = []
    for option in options.values():
        value = option.serialize()
        if value:
            serialized.append(value)
    return tuple(serialized)


def _build_compile_disk_cache_key(
    compile_callable: Any,
    func: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> str:
    payload = (
        "b12x_cute_compile_cache_v2",
        _normalize_compile_target(func, set()),
        _b12x_package_fingerprint(),
        _runtime_toolchain_key(),
        _structural_cache_key(args),
        _structural_cache_key(kwargs),
        _compile_options_cache_key(compile_callable),
        _compile_environment_key(),
    )
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()


def _cache_prefix(cache_key: str) -> str:
    return f"b12x_cute_{cache_key}"


def _cache_object_path(cache_key: str) -> Path:
    return _cute_compile_cache_dir() / cache_key[:2] / f"{cache_key}.o"


def _load_cute_compile_from_disk(cache_key: str):
    from cutlass.base_dsl.export.external_binary_module import ExternalBinaryModule

    object_path = _cache_object_path(cache_key)
    if not object_path.exists():
        return None
    try:
        module = ExternalBinaryModule(str(object_path))
        return getattr(module, _cache_prefix(cache_key))
    except Exception:
        return None


def _store_cute_compile_to_disk(cache_key: str, compiled: Any) -> None:
    if not hasattr(compiled, "dump_to_object"):
        return

    object_path = _cache_object_path(cache_key)
    object_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = object_path.with_suffix(".tmp")
    object_bytes = compiled.dump_to_object(_cache_prefix(cache_key))
    with open(tmp_path, "wb") as f:
        f.write(object_bytes)
    os.replace(tmp_path, object_path)


def _patch_cutlass_sm120_blockscaled_arch_check() -> None:
    try:
        from cutlass.cute.nvgpu.warp import mma as mma_mod
    except Exception:
        return

    blockscaled_op = getattr(mma_mod, "MmaSM120BlockScaledOp", None)
    if blockscaled_op is None:
        return
    if getattr(blockscaled_op, "_b12x_sm120a_patch", False):
        return

    original_post_init = blockscaled_op.__post_init__
    Arch = mma_mod.Arch
    dsl_base = getattr(mma_mod, "CuTeDSL", None)
    if dsl_base is None:
        dsl_base = getattr(mma_mod, "BaseDSL", None)
    if dsl_base is None:
        return
    OpError = mma_mod.OpError
    Float4E2M1FN = mma_mod.Float4E2M1FN
    Float32 = mma_mod.Float32
    Float8E4M3FN = mma_mod.Float8E4M3FN
    Float8E8M0FNU = mma_mod.Float8E8M0FNU

    @wraps(original_post_init)
    def patched_post_init(self) -> None:
        arch = dsl_base._get_dsl().get_arch_enum()
        allowed_archs = {Arch.sm_120a}
        sm_121a = getattr(Arch, "sm_121a", None)
        if sm_121a is not None:
            allowed_archs.add(sm_121a)
        if arch not in allowed_archs:
            raise OpError(
                self,
                f"expects arch to be one of {self.admissible_archs}, but got {arch}",
                suggestion="Ensure env CUTE_DSL_ARCH matches your GPU architecture",
            )
        if self.ab_dtype != Float4E2M1FN:
            raise OpError(
                self,
                "expects the 'ab_dtype' Op parameter to be Float4E2M1FN",
            )
        if self.acc_dtype != Float32:
            raise OpError(
                self,
                "expects the 'acc_dtype' Op parameter to be Float32",
            )
        if self.shape_mnk != (16, 8, 64):
            raise OpError(
                self,
                "expects the 'shape_mnk' Op parameter to be (16,8,64)",
            )
        if self.sf_vec_size == 16:
            if self.sf_type != Float8E4M3FN:
                raise OpError(
                    self,
                    "expects the 'sf_type' Op parameter to be Float8E4M3FN",
                )
        elif self.sf_vec_size == 32:
            if self.sf_type != Float8E8M0FNU:
                raise OpError(
                    self,
                    "expects the 'sf_type' Op parameter to be Float8E8M0FNU",
                )
        else:
            raise OpError(
                self,
                "expects the 'sf_vec_size' Op parameter to be 16 or 32",
            )

    blockscaled_op.__post_init__ = patched_post_init
    blockscaled_op._b12x_sm120a_patch = True


def apply_cutlass_runtime_patches() -> None:
    global _PATCHED
    if _PATCHED:
        return

    try:
        from cutlass.base_dsl.compiler import CompileCallable
        from cutlass.base_dsl.dsl import BaseDSL
        from cutlass.base_dsl.jit_executor import JitExecutor
    except Exception:
        return

    original_print_warning = BaseDSL.print_warning
    original_print_warning_once = BaseDSL.print_warning_once
    original_compile = CompileCallable._compile
    original_get_invoke_packed_args = JitExecutor._get_invoke_packed_args

    @wraps(original_print_warning)
    def patched_print_warning(self, message):
        if message == _COMPILE_ONLY_CACHE_WARNING:
            return None
        return original_print_warning(self, message)

    @wraps(original_print_warning_once)
    def patched_print_warning_once(self, message):
        if message == _COMPILE_ONLY_CACHE_WARNING:
            return None
        return original_print_warning_once(self, message)

    @wraps(original_compile)
    def patched_compile(self, func, *args, **kwargs):
        if not _cute_compile_disk_cache_enabled():
            return original_compile(self, func, *args, **kwargs)

        cache_key = _build_compile_disk_cache_key(self, func, args, kwargs)
        compiled = _load_cute_compile_from_disk(cache_key)
        if compiled is not None:
            return compiled

        compiled = original_compile(self, func, *args, **kwargs)
        with suppress(Exception):
            _store_cute_compile_to_disk(cache_key, compiled)
        return compiled

    @wraps(original_get_invoke_packed_args)
    def patched_get_invoke_packed_args(self, exe_args):
        num_base_args = len(exe_args)
        total_args = num_base_args + self._num_extra_args

        tls = self._tls
        packed_args = getattr(tls, "packed_args", None)
        capacity = getattr(tls, "capacity", 0)

        if packed_args is None or capacity < total_args:
            packed_args = (ctypes.c_void_p * total_args)()
            tls.packed_args = packed_args
            tls.capacity = total_args

            idx = num_base_args
            if self._has_cuda_result:
                packed_args[idx] = self._cuda_result_addr
                idx += 1
            if self._kernel_ptrs is not None:
                for ptr in self._kernel_ptrs:
                    packed_args[idx] = ptr
                    idx += 1

        for i, arg in enumerate(exe_args):
            if type(arg) is ctypes.c_void_p:
                packed_args[i] = arg
                continue
            try:
                packed_args[i] = ctypes.c_void_p(arg).value
            except TypeError:
                if hasattr(arg, "__int__"):
                    packed_args[i] = ctypes.c_void_p(int(arg)).value
                else:
                    raise

        return packed_args

    BaseDSL.print_warning = patched_print_warning
    BaseDSL.print_warning_once = patched_print_warning_once
    CompileCallable._compile = patched_compile
    JitExecutor._get_invoke_packed_args = patched_get_invoke_packed_args
    _patch_cutlass_sm120_blockscaled_arch_check()
    _PATCHED = True
