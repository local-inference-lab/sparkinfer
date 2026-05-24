"""Public sparse-MLA integration surface."""

from __future__ import annotations

from b12x.attention.mla import (
    MLASparseDecodeMetadata,
    MLASparseExtendMetadata,
    clear_mla_caches,
    compressed_mla_decode_forward,
    compressed_mla_split_chunks_for_contract,
    sparse_mla_decode_forward,
    sparse_mla_extend_forward,
)

__all__ = [
    "MLASparseDecodeMetadata",
    "MLASparseExtendMetadata",
    "clear_mla_caches",
    "compressed_mla_decode_forward",
    "compressed_mla_split_chunks_for_contract",
    "sparse_mla_decode_forward",
    "sparse_mla_extend_forward",
]
