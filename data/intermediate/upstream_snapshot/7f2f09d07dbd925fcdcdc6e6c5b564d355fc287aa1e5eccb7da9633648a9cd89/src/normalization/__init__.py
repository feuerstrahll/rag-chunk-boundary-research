"""Text normalization and canonical serialization."""

from .canonical import BLOCK_SEPARATOR, assign_block_offsets, build_canonical_text, validate_block_offsets

__all__ = ["BLOCK_SEPARATOR", "assign_block_offsets", "build_canonical_text", "validate_block_offsets"]
