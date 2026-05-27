"""Compressed MLA kernel planning helpers shared by runtime and scratch plans."""

from __future__ import annotations

from b12x.attention.workspace import SparseMLASplitDecodeConfig


_COMPRESSED_MLA_DECODE_SPLIT_CHUNK_SIZE = 12
_COMPRESSED_MLA_DECODE_SPLIT_MAX_ROWS = 64
_COMPRESSED_MLA_DECODE_WIDE_CHUNK_SIZE = 64
_COMPRESSED_MLA_BATCHED_SPLIT_CHUNK_SIZE = 1024
_COMPRESSED_MLA_SPLIT_MAX_CHUNKS = 256


def compressed_mla_split_config_for_contract(
    *,
    rows: int,
    width: int,
    max_chunks: int | None = None,
) -> SparseMLASplitDecodeConfig:
    rows = max(int(rows), 1)
    width = max(int(width), 1)
    chunk_limit = _COMPRESSED_MLA_SPLIT_MAX_CHUNKS
    if max_chunks is not None:
        chunk_limit = max(1, min(int(max_chunks), chunk_limit))

    decode_chunks = (
        width + _COMPRESSED_MLA_DECODE_SPLIT_CHUNK_SIZE - 1
    ) // _COMPRESSED_MLA_DECODE_SPLIT_CHUNK_SIZE
    if (
        rows <= _COMPRESSED_MLA_DECODE_SPLIT_MAX_ROWS
        and decode_chunks <= chunk_limit
    ):
        return SparseMLASplitDecodeConfig(
            chunk_size=_COMPRESSED_MLA_DECODE_SPLIT_CHUNK_SIZE,
            num_chunks=decode_chunks,
        )

    wide_decode_chunks = (
        width + _COMPRESSED_MLA_DECODE_WIDE_CHUNK_SIZE - 1
    ) // _COMPRESSED_MLA_DECODE_WIDE_CHUNK_SIZE
    if rows <= _COMPRESSED_MLA_DECODE_SPLIT_MAX_ROWS and wide_decode_chunks <= chunk_limit:
        return SparseMLASplitDecodeConfig(
            chunk_size=_COMPRESSED_MLA_DECODE_WIDE_CHUNK_SIZE,
            num_chunks=wide_decode_chunks,
        )

    chunks = (
        width + _COMPRESSED_MLA_BATCHED_SPLIT_CHUNK_SIZE - 1
    ) // _COMPRESSED_MLA_BATCHED_SPLIT_CHUNK_SIZE
    if chunks <= chunk_limit:
        return SparseMLASplitDecodeConfig(
            chunk_size=_COMPRESSED_MLA_BATCHED_SPLIT_CHUNK_SIZE,
            num_chunks=chunks,
        )

    chunk_size = (width + chunk_limit - 1) // chunk_limit
    return SparseMLASplitDecodeConfig(chunk_size=chunk_size, num_chunks=chunk_limit)


def compressed_mla_split_chunks_for_contract(
    *,
    rows: int,
    width: int,
    max_chunks: int | None = None,
) -> int:
    return compressed_mla_split_config_for_contract(
        rows=rows,
        width=width,
        max_chunks=max_chunks,
    ).num_chunks


__all__ = [
    "compressed_mla_split_chunks_for_contract",
    "compressed_mla_split_config_for_contract",
]
