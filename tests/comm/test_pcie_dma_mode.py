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
        ("AG", "ag"),
        ("ring", "ring"),
        ("a2a", "a2a"),
        ("int8", "i8"),
        ("i8-ag", "i8"),
        ("int8_ring", "i8_ring"),
        ("a2a_i8", "i8_a2a"),
        ("mxfp8", "mx"),
        ("mx-ag", "mx"),
        ("mxfp8_ring", "mx_ring"),
        ("a2a_mx", "mx_a2a"),
    ],
)
def test_supported_wire_mode_aliases(value: str, expected: str) -> None:
    assert _normalize_fp8_mode(value) == expected


def test_unknown_wire_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="unrecognized PCIe DMA wire mode"):
        _normalize_fp8_mode("mx_rnig")
