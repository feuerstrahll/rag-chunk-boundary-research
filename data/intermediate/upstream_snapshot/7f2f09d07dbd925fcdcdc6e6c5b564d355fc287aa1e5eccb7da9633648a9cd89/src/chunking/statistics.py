"""Small reusable span-coverage and fixed-chunk summary helpers."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from statistics import fmean, median
from typing import Sequence

from models import Block, Chunk


@dataclass(frozen=True)
class BlockCoverage:
    overlap_chars: int
    block_fraction: float
    fully_covered: bool


@dataclass(frozen=True)
class ChunkStatistics:
    chunk_count: int
    mean_token_count: float
    median_token_count: float
    p95_token_count: float
    min_token_count: int
    max_token_count: int
    oversize_fraction: float


def get_block_coverage(chunk: Chunk, block: Block) -> BlockCoverage:
    """Describe character overlap without calculating corpus-level integrity metrics."""
    overlap_chars = max(0, min(chunk.char_end, block.char_end) - max(chunk.char_start, block.char_start))
    block_length = block.char_end - block.char_start
    if block_length == 0:
        return BlockCoverage(overlap_chars=0, block_fraction=0.0, fully_covered=False)
    return BlockCoverage(
        overlap_chars=overlap_chars,
        block_fraction=overlap_chars / block_length,
        fully_covered=overlap_chars == block_length,
    )


def calculate_chunk_statistics(chunks: Sequence[Chunk]) -> ChunkStatistics:
    """Summarize chunks; p95 uses the deterministic nearest-rank method.

    Nearest-rank p95 is the sorted value at ``ceil(0.95 * n)`` (one-based).
    An empty sequence produces zero for every statistic.
    """
    if not chunks:
        return ChunkStatistics(0, 0.0, 0.0, 0.0, 0, 0, 0.0)

    token_counts = sorted(chunk.token_count for chunk in chunks)
    rank = ceil(0.95 * len(token_counts))
    return ChunkStatistics(
        chunk_count=len(chunks),
        mean_token_count=fmean(token_counts),
        median_token_count=float(median(token_counts)),
        p95_token_count=float(token_counts[rank - 1]),
        min_token_count=token_counts[0],
        max_token_count=token_counts[-1],
        oversize_fraction=sum(chunk.oversize for chunk in chunks) / len(chunks),
    )
