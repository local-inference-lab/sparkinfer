from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import os
import sys
import traceback
from collections import OrderedDict
from contextlib import suppress
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from threading import RLock
from types import SimpleNamespace
from typing import Any

_B12X_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_MEMORY_CACHE: OrderedDict[str, Any] = OrderedDict()
_MEMORY_CACHE_LOCK = RLock()
_MEMORY_CACHE_HITS = 0
_MEMORY_CACHE_MISSES = 0
_DISK_CACHE_HITS = 0
_COMPILE_MISSES = 0


@dataclass(frozen=True)
class DimKey:
    kind: str
    value: object = None

    @staticmethod
    def exact(value: object) -> "DimKey":
        return DimKey("exact", value)

    @staticmethod
    def capacity(value: object) -> "DimKey":
        return DimKey("capacity", value)

    @staticmethod
    def bucket(value: object) -> "DimKey":
        return DimKey("bucket", value)

    @staticmethod
    def dynamic() -> "DimKey":
        return DimKey("dynamic")

    @staticmethod
    def ignored() -> "DimKey":
        return DimKey("ignored")


@dataclass(frozen=True)
class TensorKey:
    name: str
    dtype: str
    rank: int
    dims: tuple[DimKey, ...]
    stride: tuple[int, ...]
    device: tuple[str, int | None]
    align: int | None = None
    layout: object = None

    @staticmethod
    def from_tensor(
        name: str,
        tensor: Any,
        *,
        dims: tuple[DimKey, ...] | None = None,
        align: int | None = None,
        layout: object = None,
    ) -> "TensorKey":
        shape = tuple(int(dim) for dim in tensor.shape)
        if dims is None:
            dims = tuple(DimKey.exact(dim) for dim in shape)
        if len(dims) != len(shape):
            raise ValueError(
                f"tensor key {name!r} dim policy rank {len(dims)} "
                f"does not match tensor rank {len(shape)}"
            )
        device = tensor.device
        return TensorKey(
            name=name,
            dtype=str(tensor.dtype),
            rank=len(shape),
            dims=dims,
            stride=tuple(int(stride) for stride in tensor.stride()),
            device=(device.type, device.index),
            align=align,
            layout=layout,
        )


@dataclass(frozen=True)
class KeyField:
    name: str
    value: object


@dataclass(frozen=True)
class KernelCompileSpec:
    kernel_id: str
    version: int
    fields: tuple[KeyField, ...] = ()

    @staticmethod
    def from_fields(
        kernel_id: str,
        version: int,
        *fields: KeyField | tuple[str, object],
    ) -> "KernelCompileSpec":
        return KernelCompileSpec(
            kernel_id=kernel_id,
            version=int(version),
            fields=tuple(_coerce_key_field(field) for field in fields),
        )

    @staticmethod
    def from_key(
        kernel_id: str,
        version: int,
        key: tuple[object, ...],
        *,
        labels: tuple[str, ...] | None = None,
    ) -> "KernelCompileSpec":
        if labels is not None and len(labels) != len(key):
            raise ValueError(
                f"compile spec labels length {len(labels)} does not match "
                f"key length {len(key)}"
            )
        return KernelCompileSpec(
            kernel_id=kernel_id,
            version=int(version),
            fields=tuple(
                KeyField(labels[idx] if labels is not None else f"arg{idx}", value)
                for idx, value in enumerate(key)
            ),
        )


def _coerce_key_field(field: KeyField | tuple[str, object]) -> KeyField:
    if isinstance(field, KeyField):
        return field
    name, value = field
    return KeyField(str(name), value)


def key_field(name: str, value: object) -> KeyField:
    return KeyField(name, value)


def tensor_key(
    name: str,
    tensor: Any | None,
    *,
    dims: tuple[DimKey, ...] | None = None,
    align: int | None = None,
    layout: object = None,
) -> KeyField:
    return KeyField(
        name,
        None
        if tensor is None
        else TensorKey.from_tensor(name, tensor, dims=dims, align=align, layout=layout),
    )


def _cute_compile_memory_cache_enabled() -> bool:
    raw = os.environ.get("B12X_CUTE_COMPILE_MEMORY_CACHE", "1")
    return raw.lower() not in {"0", "false", "no", ""}


def _cute_compile_memory_cache_size() -> int:
    raw = os.environ.get("B12X_CUTE_COMPILE_MEMORY_CACHE_SIZE", "1024")
    try:
        return max(1, int(raw))
    except ValueError:
        return 1024


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


def _cute_compile_log_enabled() -> bool:
    raw = os.environ.get("B12X_LOG_CUTE_COMPILES", "")
    return raw.lower() not in {"", "0", "false", "no", "off"}


def _cute_compile_stack_log_enabled() -> bool:
    raw = os.environ.get("B12X_LOG_CUTE_COMPILE_STACK", "")
    if raw:
        return raw.lower() not in {"0", "false", "no", "off"}
    raw = os.environ.get("B12X_LOG_CUTE_COMPILES", "")
    return raw.lower() in {"stack", "trace", "traceback", "full"}


def _cute_compile_stack_log_depth() -> int:
    raw = os.environ.get("B12X_LOG_CUTE_COMPILE_STACK_DEPTH", "")
    if not raw:
        return 48
    try:
        return max(1, int(raw))
    except ValueError:
        return 48


def _short_repr(value: Any, *, max_len: int = 160) -> str:
    try:
        if isinstance(value, type):
            text = f"{value.__module__}.{value.__qualname__}"
        else:
            text = repr(value)
    except Exception:
        text = f"<{type(value).__module__}.{type(value).__qualname__}>"
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def _compile_target_name(func: Any) -> str:
    unwrapped = inspect.unwrap(func)
    if inspect.ismethod(unwrapped):
        module = getattr(unwrapped.__func__, "__module__", "")
        qualname = getattr(
            unwrapped.__func__,
            "__qualname__",
            getattr(unwrapped.__func__, "__name__", ""),
        )
        return f"{module}.{qualname}" if module else qualname
    if inspect.isfunction(unwrapped):
        module = getattr(unwrapped, "__module__", "")
        qualname = getattr(unwrapped, "__qualname__", getattr(unwrapped, "__name__", ""))
        return f"{module}.{qualname}" if module else qualname
    target_type = type(func)
    module = getattr(target_type, "__module__", "")
    qualname = getattr(target_type, "__qualname__", target_type.__name__)
    return f"{module}.{qualname}" if module else qualname


def _simple_log_value(value: Any) -> Any | None:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    if isinstance(value, (tuple, list)) and len(value) <= 8:
        items = []
        for item in value:
            simple = _simple_log_value(item)
            if simple is None:
                return None
            items.append(simple)
        return tuple(items) if isinstance(value, tuple) else items
    return None


def _compile_target_attrs(func: Any) -> dict[str, Any]:
    if not hasattr(func, "__dict__"):
        return {}
    attrs = {}
    for name, value in sorted(vars(func).items()):
        if name.startswith("_"):
            continue
        simple = _simple_log_value(value)
        if simple is None:
            continue
        attrs[name] = simple
        if len(attrs) >= 48:
            attrs["..."] = "truncated"
            break
    return attrs


def _compile_arg_shape_summary(value: Any) -> dict[str, Any] | None:
    shape = _first_present_attr(value, "_shape", "shape")
    if shape is None:
        return None
    try:
        shape_value = tuple(shape)
    except TypeError:
        shape_value = shape
    summary: dict[str, Any] = {
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
        "shape": _short_repr(shape_value, max_len=96),
    }
    stride = _first_present_attr(value, "_stride", "stride")
    if stride is not None:
        try:
            stride_value = tuple(stride)
        except TypeError:
            stride_value = stride
        summary["stride"] = _short_repr(stride_value, max_len=96)
    stride_order = _first_present_attr(value, "_stride_order", "stride_order")
    if stride_order is not None:
        try:
            stride_order_value = tuple(stride_order)
        except TypeError:
            stride_order_value = stride_order
        summary["stride_order"] = _short_repr(stride_order_value, max_len=96)
    dtype = _first_present_attr(value, "_dtype", "dtype", "element_type")
    if dtype is not None:
        summary["dtype"] = _short_repr(dtype, max_len=80)
    memspace = _first_present_attr(value, "_memspace", "memspace")
    if memspace is not None:
        summary["memspace"] = _short_repr(memspace, max_len=80)
    assumed_align = _first_present_attr(value, "_assumed_align", "assumed_align")
    if assumed_align is not None:
        summary["align"] = assumed_align
    return summary


def _compile_args_shape_summary(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for idx, value in enumerate(args):
        shaped = _compile_arg_shape_summary(value)
        if shaped is not None:
            summary[f"arg{idx}"] = shaped
        elif value is None or isinstance(value, (bool, int, float, str)):
            summary[f"arg{idx}"] = value
        if len(summary) >= 24:
            summary["..."] = "truncated"
            return summary
    for name, value in sorted(kwargs.items()):
        shaped = _compile_arg_shape_summary(value)
        if shaped is not None:
            summary[f"kw:{name}"] = shaped
        elif value is None or isinstance(value, (bool, int, float, str)):
            summary[f"kw:{name}"] = value
        if len(summary) >= 24:
            summary["..."] = "truncated"
            return summary
    return summary


def _type_log_name(module: str, qualname: str) -> str:
    return f"{module}.{qualname}" if module else qualname


def _function_fingerprint_log_value(value: Any) -> Any:
    if isinstance(value, tuple) and len(value) == 3:
        module, qualname, _fingerprint = value
        if isinstance(module, str) and isinstance(qualname, str):
            return _type_log_name(module, qualname)
    return _cache_key_log_value(value, max_depth=2, max_items=8)


def _object_state_log_value(
    value: Any, *, max_depth: int, max_items: int
) -> dict[str, Any] | None:
    if not (isinstance(value, tuple) and len(value) == 4 and value[0] == "object"):
        return None
    _tag, module, qualname, attrs = value
    if not isinstance(attrs, tuple):
        return None

    state: dict[str, Any] = {}
    for idx, item in enumerate(attrs):
        if idx >= max_items:
            state["..."] = f"{len(attrs) - idx} more"
            break
        if not (isinstance(item, tuple) and len(item) == 2):
            continue
        name, attr_value = item
        state[str(name)] = _cache_key_log_value(
            attr_value, max_depth=max_depth - 1, max_items=max_items
        )
    return {"type": _type_log_name(str(module), str(qualname)), "attrs": state}


def _tensor_key_log_value(
    names: tuple[str, ...], values: tuple[Any, ...], *, max_depth: int, max_items: int
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, value in zip(names, values, strict=False):
        if value is None:
            continue
        out[name] = _cache_key_log_value(
            value, max_depth=max_depth - 1, max_items=max_items
        )
    return out


def _cache_key_log_value(
    value: Any, *, max_depth: int = 5, max_items: int = 32
) -> Any:
    if max_depth <= 0:
        return _short_repr(value, max_len=120)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, type):
        return _type_log_name(value.__module__, value.__qualname__)

    object_state = _object_state_log_value(
        value, max_depth=max_depth, max_items=max_items
    )
    if object_state is not None:
        return object_state

    if isinstance(value, tuple) and value and all(
        isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str)
        for item in value
    ):
        out: dict[str, Any] = {}
        for idx, (key, item_value) in enumerate(value):
            if idx >= max_items:
                out["..."] = f"{len(value) - idx} more"
                break
            out[key] = _cache_key_log_value(
                item_value, max_depth=max_depth - 1, max_items=max_items
            )
        return out

    if isinstance(value, tuple) and value:
        tag = value[0]
        if tag == "type" and len(value) == 3:
            return _type_log_name(str(value[1]), str(value[2]))
        if tag == "function" and len(value) == 2:
            return _function_fingerprint_log_value(value[1])
        if tag == "method" and len(value) == 3:
            return {
                "kind": "method",
                "function": _function_fingerprint_log_value(value[1]),
                "self": _cache_key_log_value(
                    value[2], max_depth=max_depth - 1, max_items=max_items
                ),
            }
        if tag == "callable_instance" and len(value) == 5:
            return {
                "kind": "callable_instance",
                "type": _type_log_name(str(value[1]), str(value[2])),
                "call": _function_fingerprint_log_value(value[3]),
                "state": _cache_key_log_value(
                    value[4], max_depth=max_depth - 1, max_items=max_items
                ),
            }
        if tag == "callable" and len(value) == 4:
            return {
                "kind": "callable",
                "type": _type_log_name(str(value[1]), str(value[2])),
                "repr": _short_repr(value[3], max_len=160),
            }
        if tag == "cache_key" and len(value) == 4:
            return {
                "type": _type_log_name(str(value[1]), str(value[2])),
                "cache_key": _cache_key_log_value(
                    value[3], max_depth=max_depth - 1, max_items=max_items
                ),
            }
        if tag == "fake_tensor" and len(value) == 12:
            return _tensor_key_log_value(
                (
                    "kind",
                    "type",
                    "dtype",
                    "shape",
                    "stride",
                    "stride_order",
                    "device",
                    "layout",
                    "memspace",
                    "align",
                    "use_32bit_stride",
                ),
                (
                    "fake_tensor",
                    _type_log_name(str(value[1]), str(value[2])),
                    *value[3:],
                ),
                max_depth=max_depth,
                max_items=max_items,
            )
        if tag == "runtime_tensor" and len(value) == 8:
            return _tensor_key_log_value(
                (
                    "kind",
                    "dtype",
                    "shape",
                    "stride",
                    "memspace",
                    "align",
                    "is_dynamic",
                    "use_32bit_stride",
                ),
                value,
                max_depth=max_depth,
                max_items=max_items,
            )
        if tag == "fake_compact_tensor" and len(value) == 7:
            return _tensor_key_log_value(
                (
                    "kind",
                    "dtype",
                    "shape",
                    "stride_order",
                    "memspace",
                    "align",
                    "use_32bit_stride",
                ),
                value,
                max_depth=max_depth,
                max_items=max_items,
            )
        if tag == "cuda_stream":
            return "cuda_stream"
        if tag == "symbolic_dim" and len(value) == 4:
            return value[3]
        if tag == "bytes" and len(value) == 2 and isinstance(value[1], str):
            return f"bytes[{len(value[1]) // 2}]"
        if tag == "path" and len(value) == 2:
            return value[1]
        if tag == "repr" and len(value) == 4:
            return {
                "type": _type_log_name(str(value[1]), str(value[2])),
                "repr": _short_repr(value[3], max_len=160),
            }
        if tag == "cycle" and len(value) == 3:
            return {"cycle": _type_log_name(str(value[1]), str(value[2]))}

    if isinstance(value, dict):
        out = {}
        for idx, (key, item_value) in enumerate(
            sorted(value.items(), key=lambda kv: str(kv[0]))
        ):
            if idx >= max_items:
                out["..."] = f"{len(value) - idx} more"
                break
            out[str(key)] = _cache_key_log_value(
                item_value, max_depth=max_depth - 1, max_items=max_items
            )
        return out

    if isinstance(value, (tuple, list)):
        items = [
            _cache_key_log_value(item, max_depth=max_depth - 1, max_items=max_items)
            for item in value[:max_items]
        ]
        if len(value) > max_items:
            items.append(f"... {len(value) - max_items} more")
        return tuple(items) if isinstance(value, tuple) else items

    return _short_repr(value, max_len=160)


def _toolchain_log_value(toolchain: tuple[object, ...]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for entry in toolchain:
        if not (isinstance(entry, tuple) and entry):
            continue
        name = entry[0]
        if name == "python" and len(entry) >= 3:
            summary[str(name)] = f"{entry[1]} {'.'.join(str(v) for v in entry[2])}"
        elif len(entry) >= 2 and entry[1]:
            summary[str(name)] = entry[1]
    return summary


def _environment_log_value(env: tuple[tuple[str, str], ...]) -> dict[str, str]:
    return {name: value for name, value in env if value}


def _compile_cache_payload_log_value(payload: tuple[object, ...] | None) -> dict[str, Any]:
    if payload is None:
        return {}
    if len(payload) == 8 and payload[0] == "b12x_cute_compile_cache_v3_explicit_spec":
        (
            _version,
            target_key,
            _b12x_fingerprint,
            toolchain_key,
            spec_key,
            kwargs_key,
            options_key,
            env_key,
        ) = payload

        summary: dict[str, Any] = {
            "target": _cache_key_log_value(target_key, max_depth=7, max_items=80),
            "spec": _cache_key_log_value(spec_key, max_depth=7, max_items=80),
        }
        if options_key:
            summary["options"] = _cache_key_log_value(
                options_key, max_depth=4, max_items=32
            )
        kwargs_summary = _cache_key_log_value(kwargs_key, max_depth=5, max_items=32)
        if kwargs_summary:
            summary["kwargs"] = kwargs_summary
        env_summary = (
            _environment_log_value(env_key) if isinstance(env_key, tuple) else {}
        )
        if env_summary:
            summary["env"] = env_summary
        if isinstance(toolchain_key, tuple):
            toolchain_summary = _toolchain_log_value(toolchain_key)
            if toolchain_summary:
                summary["toolchain"] = toolchain_summary
        return summary

    if len(payload) != 8:
        return {}
    (
        _version,
        target_key,
        _b12x_fingerprint,
        toolchain_key,
        args_key,
        kwargs_key,
        options_key,
        env_key,
    ) = payload

    summary: dict[str, Any] = {
        "target": _cache_key_log_value(target_key, max_depth=7, max_items=80),
        "args": _cache_key_log_value(args_key, max_depth=5, max_items=32),
    }
    kwargs_summary = _cache_key_log_value(kwargs_key, max_depth=5, max_items=32)
    if kwargs_summary:
        summary["kwargs"] = kwargs_summary
    if options_key:
        summary["options"] = _cache_key_log_value(
            options_key, max_depth=4, max_items=32
        )
    env_summary = _environment_log_value(env_key) if isinstance(env_key, tuple) else {}
    if env_summary:
        summary["env"] = env_summary
    if isinstance(toolchain_key, tuple):
        toolchain_summary = _toolchain_log_value(toolchain_key)
        if toolchain_summary:
            summary["toolchain"] = toolchain_summary
    return summary


def _format_cute_compile_stack() -> str:
    depth = _cute_compile_stack_log_depth()
    frames = traceback.extract_stack()[:-2]
    runtime_patch_path = str(Path(__file__).resolve())
    visible_frames = [
        frame
        for frame in frames
        if str(Path(frame.filename).resolve()) != runtime_patch_path
    ]
    if depth:
        visible_frames = visible_frames[-depth:]

    lines = ["[b12x cute.compile] python_stack (most recent call last):"]
    for frame in visible_frames:
        lines.append(
            f'  File "{frame.filename}", line {frame.lineno}, in {frame.name}'
        )
        if frame.line:
            lines.append(f"    {frame.line.strip()}")
    return "\n".join(lines)


def _log_cute_compile_miss(
    func: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    cache_status: str,
    cache_payload: tuple[object, ...] | None = None,
) -> None:
    key_inputs = _compile_cache_payload_log_value(cache_payload)
    print(
        "[b12x cute.compile] miss "
        f"target={_compile_target_name(func)} "
        f"status={cache_status} "
        f"attrs={_short_repr(_compile_target_attrs(func), max_len=1200)} "
        f"args={_short_repr(_compile_args_shape_summary(args, kwargs), max_len=1600)} "
        f"key_inputs={_short_repr(key_inputs, max_len=4000)}",
        flush=True,
    )
    if _cute_compile_stack_log_enabled():
        print(_format_cute_compile_stack(), flush=True)


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


def _compute_b12x_package_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in _iter_fingerprint_files(_B12X_PACKAGE_ROOT):
        rel_path = str(path.relative_to(_B12X_PACKAGE_ROOT))
        digest.update(rel_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


@lru_cache(maxsize=1)
def _b12x_package_fingerprint() -> str:
    return _compute_b12x_package_fingerprint()


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


@lru_cache(maxsize=1)
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


@lru_cache(maxsize=16)
def _static_compile_cache_context(compile_callable: Any) -> tuple[object, ...]:
    return (
        _b12x_package_fingerprint(),
        _runtime_toolchain_key(),
        _compile_options_cache_key(compile_callable),
        _compile_environment_key(),
    )


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


def _explicit_spec_compile_target(func: Any) -> Any:
    if inspect.ismethod(func):
        return ("method", _function_fingerprint(func.__func__))
    if inspect.isfunction(func):
        return ("function", _function_fingerprint(func))
    if callable(func) and hasattr(func.__call__, "__func__"):
        return (
            "callable_instance",
            type(func).__module__,
            type(func).__qualname__,
            _function_fingerprint(func.__call__.__func__),
        )
    return ("callable", type(func).__module__, type(func).__qualname__)


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


def _compile_disk_cache_payload(
    compile_callable: Any,
    func: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    compile_spec: KernelCompileSpec | None = None,
) -> tuple[object, ...]:
    (
        package_fingerprint,
        runtime_toolchain,
        compile_options,
        compile_environment,
    ) = _static_compile_cache_context(compile_callable)
    if compile_spec is not None:
        return (
            "b12x_cute_compile_cache_v3_explicit_spec",
            _explicit_spec_compile_target(func),
            package_fingerprint,
            runtime_toolchain,
            _structural_cache_key(compile_spec),
            _structural_cache_key(kwargs),
            compile_options,
            compile_environment,
        )
    return (
        "b12x_cute_compile_cache_v2",
        _normalize_compile_target(func, set()),
        package_fingerprint,
        runtime_toolchain,
        _structural_cache_key(args),
        _structural_cache_key(kwargs),
        compile_options,
        compile_environment,
    )


def _build_compile_disk_cache_key(
    compile_callable: Any,
    func: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    compile_spec: KernelCompileSpec | None = None,
) -> str:
    payload = _compile_disk_cache_payload(
        compile_callable, func, args, kwargs, compile_spec
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


def _memory_cache_get(cache_key: str) -> Any | None:
    global _MEMORY_CACHE_HITS
    global _MEMORY_CACHE_MISSES
    if not _cute_compile_memory_cache_enabled():
        return None
    with _MEMORY_CACHE_LOCK:
        compiled = _MEMORY_CACHE.get(cache_key)
        if compiled is None:
            _MEMORY_CACHE_MISSES += 1
            return None
        _MEMORY_CACHE_HITS += 1
        _MEMORY_CACHE.move_to_end(cache_key)
        return compiled


def _memory_cache_put(cache_key: str, compiled: Any) -> None:
    if not _cute_compile_memory_cache_enabled():
        return
    with _MEMORY_CACHE_LOCK:
        _MEMORY_CACHE[cache_key] = compiled
        _MEMORY_CACHE.move_to_end(cache_key)
        while len(_MEMORY_CACHE) > _cute_compile_memory_cache_size():
            _MEMORY_CACHE.popitem(last=False)


def clear_compile_cache() -> None:
    global _MEMORY_CACHE_HITS
    global _MEMORY_CACHE_MISSES
    global _DISK_CACHE_HITS
    global _COMPILE_MISSES
    _compile_environment_key.cache_clear()
    _static_compile_cache_context.cache_clear()
    with _MEMORY_CACHE_LOCK:
        _MEMORY_CACHE.clear()
        _MEMORY_CACHE_HITS = 0
        _MEMORY_CACHE_MISSES = 0
        _DISK_CACHE_HITS = 0
        _COMPILE_MISSES = 0


def compile_cache_info() -> dict[str, int | bool]:
    with _MEMORY_CACHE_LOCK:
        return {
            "memory_cache_enabled": _cute_compile_memory_cache_enabled(),
            "memory_cache_size": len(_MEMORY_CACHE),
            "memory_cache_max_size": _cute_compile_memory_cache_size(),
            "memory_cache_hits": _MEMORY_CACHE_HITS,
            "memory_cache_misses": _MEMORY_CACHE_MISSES,
            "disk_cache_enabled": _cute_compile_disk_cache_enabled(),
            "disk_cache_hits": _DISK_CACHE_HITS,
            "compile_misses": _COMPILE_MISSES,
        }


def compile(
    func: Any,
    *args: Any,
    compile_spec: KernelCompileSpec | None = None,
    **kwargs: Any,
) -> Any:
    import cutlass.cute as cute

    global _DISK_CACHE_HITS
    global _COMPILE_MISSES
    compile_callable = cute.compile
    payload = _compile_disk_cache_payload(
        compile_callable, func, args, kwargs, compile_spec
    )
    cache_key = hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()

    compiled = _memory_cache_get(cache_key)
    if compiled is not None:
        return compiled

    if _cute_compile_disk_cache_enabled():
        compiled = _load_cute_compile_from_disk(cache_key)
        if compiled is not None:
            with _MEMORY_CACHE_LOCK:
                _DISK_CACHE_HITS += 1
            _memory_cache_put(cache_key, compiled)
            return compiled
        cache_status = "disk-cache-miss"
    else:
        cache_status = "disk-cache-disabled"

    if _cute_compile_log_enabled():
        with suppress(Exception):
            _log_cute_compile_miss(
                func,
                args,
                kwargs,
                cache_status=cache_status,
                cache_payload=payload,
            )

    with _MEMORY_CACHE_LOCK:
        _COMPILE_MISSES += 1
    from b12x.cute.runtime_control import raise_if_kernel_resolution_frozen

    raise_if_kernel_resolution_frozen(
        "cute.compile",
        target=func,
        cache_key=compile_spec if compile_spec is not None else payload,
    )
    compiled = compile_callable(func, *args, **kwargs)
    if _cute_compile_disk_cache_enabled():
        with suppress(Exception):
            _store_cute_compile_to_disk(cache_key, compiled)
    _memory_cache_put(cache_key, compiled)
    return compiled


def run_compiled(compiled: Any, args: tuple[Any, ...]) -> Any:
    if hasattr(compiled, "generate_execution_args") and hasattr(
        compiled, "run_compiled_program"
    ):
        execution_args, _ = compiled.generate_execution_args(*args)
        return compiled.run_compiled_program(execution_args)
    return compiled(*args)


def launch(
    func: Any,
    *,
    compile_spec: KernelCompileSpec,
    compile_args: tuple[Any, ...],
    runtime_args: tuple[Any, ...],
    compile_kwargs: dict[str, Any] | None = None,
) -> Any:
    compiled = compile(
        func,
        *compile_args,
        compile_spec=compile_spec,
        **(compile_kwargs or {}),
    )
    return run_compiled(compiled, runtime_args)
