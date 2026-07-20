#!/usr/bin/env python3
"""Fail-closed checks for exact per-nodeid corpus GPU execution proof.

This is an offline self-test of the Nsight/NVTX evidence validator, not a CPU
kernel test and not migration acceptance evidence. Release evidence still
requires real execution on physical GPUs 4 and 5.
"""

from __future__ import annotations

import sqlite3
import tempfile
from collections.abc import Callable
from pathlib import Path

from validation.cutlass_migration.acceptance.corpus.run import (
    CorpusError,
    _nsys_trace,
    _test_nvtx_label,
)


_CASE_ID = "integrity-gpu-execution"
_NODEIDS = (
    "tests/test_bf16_to_fp4_tma.py::test_name_alone_cannot_prove_gpu_a",
    "tests/test_bf16_to_fp4_tma.py::test_name_alone_cannot_prove_gpu_b",
)
_CASE_RANGE = (100, 1_000)
_TEST_RANGES = ((200, 400), (500, 700))


def _expected_ranges() -> list[dict[str, object]]:
    return [
        {
            "nodeid": nodeid,
            "label": _test_nvtx_label(_CASE_ID, nodeid),
            "completed": True,
        }
        for nodeid in _NODEIDS
    ]


def _write_trace(
    path: Path,
    *,
    kernels: tuple[tuple[int, int, str], ...],
    included_test_labels: tuple[int, ...] = (0, 1),
) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute("CREATE TABLE StringIds (id INTEGER, value TEXT)")
        connection.execute(
            "CREATE TABLE NVTX_EVENTS (start INTEGER, end INTEGER, text TEXT)"
        )
        connection.execute(
            "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL "
            "(start INTEGER, end INTEGER, shortName TEXT, deviceId INTEGER, "
            "registersPerThread INTEGER)"
        )
        connection.execute(
            "INSERT INTO NVTX_EVENTS VALUES (?, ?, ?)",
            (*_CASE_RANGE, f"sparkinfer-cute-corpus-case:{_CASE_ID}"),
        )
        for index in included_test_labels:
            connection.execute(
                "INSERT INTO NVTX_EVENTS VALUES (?, ?, ?)",
                (
                    *_TEST_RANGES[index],
                    _test_nvtx_label(_CASE_ID, _NODEIDS[index]),
                ),
            )
        connection.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, 0, 64)",
            kernels,
        )
        connection.commit()
    finally:
        connection.close()


def _trace(
    *,
    kernels: tuple[tuple[int, int, str], ...],
    included_test_labels: tuple[int, ...] = (0, 1),
) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="sparkinfer-corpus-gpu-proof-") as directory:
        path = Path(directory) / "trace.sqlite"
        _write_trace(
            path,
            kernels=kernels,
            included_test_labels=included_test_labels,
        )
        return _nsys_trace(path, _CASE_ID, _expected_ranges())


def _expect_failure(callback: Callable[[], object], message: str) -> None:
    try:
        callback()
    except CorpusError as error:
        if message not in str(error):
            raise AssertionError(
                f"expected failure containing {message!r}, got {str(error)!r}"
            ) from error
    else:
        raise AssertionError(f"invalid GPU execution evidence passed: {message}")


def main() -> int:
    valid = _trace(
        kernels=((250, 300, "kernel_a"), (550, 600, "kernel_b")),
    )
    test_ranges = valid.get("test_ranges")
    if not isinstance(test_ranges, dict) or set(test_ranges) != set(_NODEIDS):
        raise AssertionError("valid trace lost exact pytest-nodeid ownership")
    for nodeid in _NODEIDS:
        record = test_ranges[nodeid]
        if not isinstance(record, dict) or len(record.get("cuda_events", [])) != 1:
            raise AssertionError(f"valid trace lacks one CUDA event for {nodeid}")

    # Both nodeids look GPU-oriented. Omitting the second launch must still
    # fail, proving that names and CUDA-looking source cannot satisfy the gate.
    _expect_failure(
        lambda: _trace(kernels=((250, 300, "kernel_a"),)),
        "GPU-only corpus tests have no CUDA launches in their exact ranges",
    )
    _expect_failure(
        lambda: _trace(kernels=((250, 300, "kernel_a"), (390, 410, "crossing_kernel"))),
        "crosses or ambiguously occupies test ranges",
    )
    _expect_failure(
        lambda: _trace(
            kernels=((250, 300, "kernel_a"), (550, 600, "kernel_b")),
            included_test_labels=(0,),
        ),
        "test NVTX label mismatch",
    )

    print(
        "status=pass proof=nsys-exact-per-test-nvtx-cuda-event "
        "positive=exact-nodeid-ownership "
        "negative=empty-test-range,cross-boundary,missing-test-range"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
