"""Deterministic canonical serialization for normalized document blocks."""

from __future__ import annotations

from models import Document

BLOCK_SEPARATOR = "\n\n"


def build_canonical_text(document: Document) -> str:
    """Serialize normalized blocks without owning a second mutable text copy."""
    return BLOCK_SEPARATOR.join(block.text for block in document.blocks)


def assign_block_offsets(document: Document) -> None:
    """Assign monotonically increasing, non-overlapping offsets after normalization."""
    cursor = 0
    for index, block in enumerate(document.blocks):
        block.char_end = cursor + len(block.text)
        block.char_start = cursor
        cursor = block.char_end
        if index < len(document.blocks) - 1:
            cursor += len(BLOCK_SEPARATOR)
    validate_block_offsets(document)


def validate_block_offsets(document: Document) -> None:
    """Confirm every block round-trips through the derived canonical text."""
    canonical_text = build_canonical_text(document)
    previous_end = 0
    for index, block in enumerate(document.blocks):
        if block.char_start < previous_end:
            raise ValueError(f"Overlapping block offsets: {block.block_id}")
        if canonical_text[block.char_start : block.char_end] != block.text:
            raise ValueError(f"Canonical text does not match block: {block.block_id}")
        previous_end = block.char_end
        if index < len(document.blocks) - 1:
            previous_end += len(BLOCK_SEPARATOR)
