"""Tests for PCIe-DMA wire-mode parsing."""

from __future__ import annotations

import pytest

from sparkinfer.comm.pcie.pcie_dma import _normalize_fp8_mode


@pytest.mark.parametrize("value", [None, "", "0", "false", "off", "no"])
def test_disabled_wire_mode_aliases(value: str | None) -> None:
    assert _normalize_fp8_mode(value) == ""


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1", "ag"),
        (" AG ", "ag"),
        ("ring", "ring"),
        ("a2a", "a2a"),
        ("i8", "i8"),
        ("int8", "i8"),
        ("i8_ag", "i8"),
        ("i8-ag", "i8"),
        ("ag_i8", "i8"),
        ("int8_ag", "i8"),
        ("int8-ag", "i8"),
        ("i8_ring", "i8_ring"),
        ("i8-ring", "i8_ring"),
        ("int8_ring", "i8_ring"),
        ("int8-ring", "i8_ring"),
        ("ring_i8", "i8_ring"),
        ("i8_a2a", "i8_a2a"),
        ("i8-a2a", "i8_a2a"),
        ("int8_a2a", "i8_a2a"),
        ("int8-a2a", "i8_a2a"),
        ("a2a_i8", "i8_a2a"),
        ("  InT8-RiNg ", "i8_ring"),
        ("mx", "mx"),
        ("mxfp8", "mx"),
        ("mx_ag", "mx"),
        ("mx-ag", "mx"),
        ("mxfp8_ag", "mx"),
        ("mxfp8-ag", "mx"),
        ("ag_mx", "mx"),
        ("mx_ring", "mx_ring"),
        ("mx-ring", "mx_ring"),
        ("mxfp8_ring", "mx_ring"),
        ("mxfp8-ring", "mx_ring"),
        ("ring_mx", "mx_ring"),
        ("mx_a2a", "mx_a2a"),
        ("mx-a2a", "mx_a2a"),
        ("mxfp8_a2a", "mx_a2a"),
        ("mxfp8-a2a", "mx_a2a"),
        ("a2a_mx", "mx_a2a"),
        (" MxFp8-RiNg ", "mx_ring"),
    ],
)
def test_supported_wire_mode_aliases(value: str, expected: str) -> None:
    assert _normalize_fp8_mode(value) == expected


def test_unknown_wire_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="unrecognized PCIe DMA wire mode"):
        _normalize_fp8_mode("mx_rnig")
