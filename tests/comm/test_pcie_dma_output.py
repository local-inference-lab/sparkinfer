from __future__ import annotations

import pytest
import torch

from sparkinfer.comm.pcie.pcie_dma import (
    OUTPUT_TAIL_PADDING,
    _align_up,
    _persistent_output_view,
)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_persistent_output_view_reuses_storage_with_mapped_tail(
    dtype: torch.dtype,
) -> None:
    max_bytes = 4096
    storage = torch.empty(
        _align_up(max_bytes, 8) + OUTPUT_TAIL_PADDING,
        dtype=torch.uint8,
    )
    elements = max_bytes // dtype.itemsize
    inp = torch.empty((16, elements // 16), dtype=dtype)

    output = _persistent_output_view(storage, inp, max_bytes)
    smaller = _persistent_output_view(storage, inp[:1], max_bytes)

    assert output.shape == inp.shape
    assert output.dtype == dtype
    assert output.is_contiguous()
    assert output.data_ptr() == storage.data_ptr()
    assert smaller.data_ptr() == output.data_ptr()
    assert (
        storage.data_ptr() + storage.numel() - output.data_ptr() - max_bytes
        >= OUTPUT_TAIL_PADDING
    )


def test_persistent_output_view_rejects_capacity_overflow() -> None:
    storage = torch.empty(128 + OUTPUT_TAIL_PADDING, dtype=torch.uint8)
    inp = torch.empty(65, dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="capacity is 128"):
        _persistent_output_view(storage, inp, 128)


def test_persistent_output_view_rejects_wrong_storage() -> None:
    inp = torch.empty(16, dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="uint8 on the input device"):
        _persistent_output_view(torch.empty(128), inp, 128)
