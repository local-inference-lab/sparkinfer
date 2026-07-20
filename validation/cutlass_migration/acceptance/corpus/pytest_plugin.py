"""GPU-only pytest telemetry for the CUTLASS migration corpus.

This plugin is loaded only by the ``python -m validation.cutlass_migration
acceptance corpus`` runner.  It records the in-process sparkinfer compile-cache
counters and the exact pytest outcomes, so a passing subprocess cannot hide
skips or a zero-kernel/reference-only run.
"""

from __future__ import annotations

from collections.abc import Generator
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

from validation.cutlass_migration.paths import REPO_ROOT


_REPORTS: list[dict[str, Any]] = []
_GPU: dict[str, Any] = {}
_TEST_NVTX_RANGES: dict[str, dict[str, Any]] = {}
_SESSION_NVTX_OPEN = False
_SOURCE_ATTESTATION_START: dict[str, Any] = {}


def pytest_configure(config: pytest.Config) -> None:
    del config
    from validation.cutlass_migration.acceptance.corpus.source_snapshot import (
        verify_frozen_source_from_environment,
    )
    from validation.cutlass_migration.acceptance.corpus.ptx_capture import (
        installation_status,
    )

    _SOURCE_ATTESTATION_START.update(
        verify_frozen_source_from_environment(
            repo_root=REPO_ROOT,
            stage="pytest_pre_collection",
        )
    )
    status = installation_status()
    if status.get("enabled") is True and status.get("installed") is not True:
        raise pytest.UsageError(
            "frontend PTX capture must be installed by the corpus launcher "
            "before pytest imports the runtime"
        )


def _telemetry_path() -> Path:
    raw = os.environ.get("CORPUS_TELEMETRY")
    if not raw:
        raise RuntimeError("CORPUS_TELEMETRY is required")
    return Path(raw)


def _test_nvtx_label(case_id: str, nodeid: str) -> str:
    identity = hashlib.sha256(f"{case_id}\0{nodeid}".encode()).hexdigest()
    return f"sparkinfer-cute-corpus-test:{case_id}:{identity}"


def pytest_sessionstart(session: pytest.Session) -> None:
    del session
    import torch

    if not torch.cuda.is_available():
        raise pytest.UsageError("CUTLASS migration corpus requires a CUDA GPU")
    if torch.cuda.device_count() != 1:
        raise pytest.UsageError(
            "corpus subprocess must expose exactly one physical GPU through "
            f"CUDA_VISIBLE_DEVICES; found {torch.cuda.device_count()}"
        )
    ordinal = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(ordinal)
    physical_ordinal = int(os.environ["CORPUS_PHYSICAL_GPU"])
    nvidia_smi_uuid = os.environ.get("CORPUS_EXPECTED_GPU_UUID", "").strip()
    nvidia_smi_name = os.environ.get("CORPUS_EXPECTED_GPU_NAME", "").strip()
    if not nvidia_smi_uuid or not nvidia_smi_name:
        raise pytest.UsageError(
            "corpus runner must provide its independently queried GPU UUID and name"
        )
    if str(props.name) != nvidia_smi_name:
        raise pytest.UsageError(
            "CUDA-visible device name differs from the runner's nvidia-smi identity: "
            f"torch={str(props.name)!r} nvidia-smi={nvidia_smi_name!r}"
        )
    cuda_uuid = str(props.uuid).lower()
    expected_cuda_uuid = nvidia_smi_uuid.removeprefix("GPU-").lower()
    if cuda_uuid != expected_cuda_uuid:
        raise pytest.UsageError(
            "CUDA-visible device UUID differs from the runner's independently "
            "queried nvidia-smi identity: "
            f"torch={cuda_uuid!r} nvidia-smi={nvidia_smi_uuid!r}"
        )
    capability = tuple(
        int(value) for value in torch.cuda.get_device_capability(ordinal)
    )
    if capability != (12, 0):
        raise pytest.UsageError(
            f"CUTLASS migration corpus requires SM120, found {capability}"
        )
    _GPU.update(
        {
            "visible_ordinal": ordinal,
            "physical_ordinal": physical_ordinal,
            "name": str(props.name),
            "nvidia_smi_name": nvidia_smi_name,
            "capability": list(capability),
            "uuid": f"GPU-{cuda_uuid}",
            "nvidia_smi_uuid": nvidia_smi_uuid,
            "total_memory": int(props.total_memory),
            "multi_processor_count": int(props.multi_processor_count),
        }
    )
    global _SESSION_NVTX_OPEN
    case_id = os.environ["CORPUS_CASE_ID"]
    torch.cuda.nvtx.range_push(f"sparkinfer-cute-corpus-case:{case_id}")
    _SESSION_NVTX_OPEN = True


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item: pytest.Item) -> Generator[None, None, None]:
    import torch

    case_id = os.environ["CORPUS_CASE_ID"]
    nodeid = str(item.nodeid)
    if nodeid in _TEST_NVTX_RANGES:
        raise pytest.UsageError(f"duplicate corpus pytest nodeid {nodeid!r}")
    label = _test_nvtx_label(case_id, nodeid)
    _TEST_NVTX_RANGES[nodeid] = {
        "nodeid": nodeid,
        "label": label,
        "completed": False,
    }
    torch.cuda.nvtx.range_push(label)
    try:
        yield
    finally:
        # CUDA work is asynchronous.  Completing it before closing the range
        # makes range containment an exact launch proof instead of a launch
        # enqueue heuristic.
        synchronized = False
        try:
            torch.cuda.synchronize()
            synchronized = True
        finally:
            torch.cuda.nvtx.range_pop()
            _TEST_NVTX_RANGES[nodeid]["completed"] = synchronized


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo[Any]
) -> Generator[None, None, None]:
    outcome = yield
    report = outcome.get_result()
    _REPORTS.append(
        {
            "nodeid": str(item.nodeid),
            "when": str(report.when),
            "outcome": str(report.outcome),
            "duration_s": float(report.duration),
            "wasxfail": str(getattr(report, "wasxfail", "")),
            "longrepr": str(report.longrepr) if report.failed else "",
        }
    )


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    import torch

    from sparkinfer.cute.compiler import compile_cache_info
    from validation.cutlass_migration.acceptance.corpus.ptx_capture import validate_cache
    from validation.cutlass_migration.acceptance.corpus.source_snapshot import (
        verify_frozen_source_from_environment,
    )

    global _SESSION_NVTX_OPEN
    if _SESSION_NVTX_OPEN:
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
        _SESSION_NVTX_OPEN = False

    cache_dir = Path(os.environ["SPARKINFER_CUTE_COMPILE_CACHE_DIR"])
    # nvdisasm subprocesses launched while Nsight/CUPTI is attached can hang
    # after completing their work.  The unprofiled prewarm and final
    # coordinator perform exact cubin redisassembly; the profiled child still
    # validates every immutable PTX/cubin/manifest identity without spawning
    # tools inside the profiler.
    redisassemble = os.environ.get("CORPUS_UNDER_NSYS") != "1"
    ptx_capture = validate_cache(cache_dir, redisassemble=redisassemble)
    if (
        ptx_capture.get("status") == "error"
        or ptx_capture.get("redisassembled") is not redisassemble
    ):
        session.exitstatus = pytest.ExitCode.INTERNAL_ERROR
        exitstatus = int(session.exitstatus)

    source_session_finish = verify_frozen_source_from_environment(
        repo_root=REPO_ROOT,
        stage="pytest_session_finish",
    )

    payload = {
        "schema": "sparkinfer.cute.migration.pytest_telemetry.v3",
        "exitstatus": int(exitstatus),
        "gpu": dict(_GPU),
        "compile_cache": compile_cache_info(),
        "frontend_ptx_capture": ptx_capture,
        "source_attestation": {
            "pre_collection": dict(_SOURCE_ATTESTATION_START),
            "session_finish": source_session_finish,
        },
        "reports": list(_REPORTS),
        "test_nvtx_ranges": [
            dict(_TEST_NVTX_RANGES[nodeid]) for nodeid in sorted(_TEST_NVTX_RANGES)
        ],
    }
    path = _telemetry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
