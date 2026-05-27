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
from b12x.integration.compressed_scratch import (
    B12XCompressedMLABinding,
    B12XCompressedMLAScratch,
    B12XCompressedMLAScratchCaps,
    B12XCompressedMLAScratchPlan,
    plan_compressed_mla_scratch,
)
from b12x.integration.sparse_mla_scratch import (
    B12XSparseMLABinding,
    B12XSparseMLAScratchCaps,
    B12XSparseMLAScratchPlan,
    plan_sparse_mla_scratch,
)

__all__ = [
    "B12XCompressedMLABinding",
    "B12XCompressedMLAScratch",
    "B12XCompressedMLAScratchCaps",
    "B12XCompressedMLAScratchPlan",
    "B12XSparseMLABinding",
    "B12XSparseMLAScratchCaps",
    "B12XSparseMLAScratchPlan",
    "MLASparseDecodeMetadata",
    "MLASparseExtendMetadata",
    "clear_mla_caches",
    "compressed_mla_decode_forward",
    "compressed_mla_split_chunks_for_contract",
    "plan_compressed_mla_scratch",
    "plan_sparse_mla_scratch",
    "sparse_mla_decode_forward",
    "sparse_mla_extend_forward",
]
