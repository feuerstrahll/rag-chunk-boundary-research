"""Deterministic fixed-token chunking with canonical-text provenance."""

from __future__ import annotations

import re
import warnings
from bisect import bisect_left
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Mapping, Sequence

import orjson
from pydantic import BaseModel, ConfigDict, Field, model_validator
from transformers import AutoTokenizer

from models import Block, Chunk, Document
from normalization import validate_block_offsets


@dataclass(frozen=True)
class TokenWindow:
    """An internal token range and its exact canonical character range."""

    token_start: int
    token_end: int
    char_start: int
    char_end: int


class TokenizerRevisionWarning(UserWarning):
    """The tokenizer works, but its exact source revision is unavailable."""


class TokenizerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    revision: str | None = None


class FixedChunkingConfig(BaseModel):
    """Resolved configuration required by the fixed-token chunker."""

    model_config = ConfigDict(extra="forbid")

    strategy: str = "fixed"
    tokenizer: TokenizerConfig
    max_tokens: int
    overlap_tokens: int

    @model_validator(mode="after")
    def validate_fixed_window(self) -> "FixedChunkingConfig":
        if self.strategy != "fixed":
            raise ValueError("FixedTokenChunker requires chunking.strategy='fixed'")
        if self.max_tokens <= 0:
            raise ValueError("chunking.max_tokens must be greater than zero")
        if not 0 <= self.overlap_tokens < self.max_tokens:
            raise ValueError("chunking.overlap_tokens must satisfy 0 <= overlap_tokens < max_tokens")
        return self

    @property
    def stride(self) -> int:
        return self.max_tokens - self.overlap_tokens

    @classmethod
    def from_resolved_config(cls, config: Mapping[str, Any]) -> "FixedChunkingConfig":
        """Read only the resolved ``chunking`` section of a pipeline config."""
        chunking = config.get("chunking")
        if not isinstance(chunking, Mapping):
            raise ValueError("Resolved configuration must contain a chunking mapping")
        return cls.model_validate(chunking)


def validate_chunking_document(document: Document) -> None:
    """Require blocks to round-trip through final canonical text before chunking."""
    validate_block_offsets(document)


def canonical_text_hash(document: Document) -> str:
    return sha256(document.canonical_text.encode("utf-8")).hexdigest()


def _snapshot_revision(value: Any) -> str | None:
    if isinstance(value, str):
        match = re.search(r"(?:^|[/\\])snapshots[/\\]([0-9a-fA-F]{7,64})(?:[/\\]|$)", value)
        return match.group(1).lower() if match else None
    if isinstance(value, Mapping):
        for nested in value.values():
            revision = _snapshot_revision(nested)
            if revision:
                return revision
    if isinstance(value, (list, tuple)):
        for nested in value:
            revision = _snapshot_revision(nested)
            if revision:
                return revision
    return None


def _resolve_tokenizer_revision(tokenizer: Any, configured: TokenizerConfig) -> str | None:
    """Resolve an exact cached Hugging Face snapshot revision when available."""
    for candidate in (
        getattr(tokenizer, "_commit_hash", None),
        getattr(tokenizer, "commit_hash", None),
        _snapshot_revision(getattr(tokenizer, "init_kwargs", {})),
    ):
        if isinstance(candidate, str) and candidate:
            return candidate

    try:
        from huggingface_hub import try_to_load_from_cache

        cached_file = try_to_load_from_cache(
            configured.name,
            "tokenizer_config.json",
            revision=configured.revision or "main",
        )
        revision = _snapshot_revision(cached_file)
        if revision:
            return revision
    except (ImportError, OSError, ValueError):
        pass

    if configured.revision is not None:
        return configured.revision
    warnings.warn(
        f"Could not resolve an exact Hugging Face revision for tokenizer {configured.name!r}; "
        "chunk IDs record an unresolved revision and reproducibility is not guaranteed.",
        TokenizerRevisionWarning,
        stacklevel=3,
    )
    return None


def _chunker_config_hash(config: FixedChunkingConfig, resolved_revision: str | None) -> str:
    canonical_config = {
        "window_algorithm": "retokenized-v2",
        "max_tokens": config.max_tokens,
        "overlap_tokens": config.overlap_tokens,
        "resolved_tokenizer_revision": resolved_revision,
        "strategy": config.strategy,
        "tokenizer_name": config.tokenizer.name,
    }
    canonical_bytes = orjson.dumps(canonical_config, option=orjson.OPT_SORT_KEYS)
    return sha256(canonical_bytes).hexdigest()


def _chunk_id(
    *, doc_id: str, text_hash: str, config_hash: str, char_start: int, char_end: int
) -> str:
    identity = orjson.dumps([doc_id, text_hash, config_hash, char_start, char_end])
    return f"chk_{sha256(identity).hexdigest()[:32]}"


def _validate_offset_mapping(canonical_text: str, offsets: Sequence[Sequence[int]]) -> None:
    previous_start = 0
    previous_end = 0
    for index, offset in enumerate(offsets):
        if len(offset) != 2:
            raise ValueError(f"Tokenizer returned an invalid offset_mapping entry at token {index}")
        start, end = offset
        if not isinstance(start, int) or not isinstance(end, int):
            raise ValueError(f"Tokenizer returned non-integer offsets at token {index}")
        if not 0 <= start <= end <= len(canonical_text):
            raise ValueError(f"Tokenizer offsets cannot be mapped to Python string slicing at token {index}")
        if start < previous_start or end < previous_end:
            raise ValueError(f"Tokenizer returned non-monotonic offsets at token {index}")
        previous_start, previous_end = start, end


def _longest_common_prefix(paths: Sequence[tuple[str, ...]]) -> tuple[str, ...]:
    if not paths:
        return ()
    prefix = paths[0]
    for path in paths[1:]:
        length = min(len(prefix), len(path))
        index = 0
        while index < length and prefix[index] == path[index]:
            index += 1
        prefix = prefix[:index]
        if not prefix:
            break
    return prefix


def _provenance_for_span(
    blocks: Sequence[Block], block_starts: Sequence[int], char_start: int, char_end: int
) -> tuple[tuple[str, ...], tuple[int, ...], tuple[str, ...], tuple[str, ...]]:
    """Derive block provenance by scanning only blocks near the given span."""
    first = bisect_left(block_starts, char_start)
    if first:
        first -= 1

    contributing: list[Block] = []
    for block in blocks[first:]:
        if block.char_start >= char_end:
            break
        if block.char_start < char_end and block.char_end > char_start:
            contributing.append(block)

    block_ids = tuple(dict.fromkeys(block.block_id for block in contributing))
    pages = tuple(sorted({block.page for block in contributing if block.page is not None}))
    block_types = tuple(dict.fromkeys(block.type for block in contributing))
    section_path = _longest_common_prefix([block.section_path for block in contributing])
    return block_ids, pages, block_types, section_path


class FixedTokenChunker:
    """Split all canonical text into overlapping tokenizer windows."""

    def __init__(self, config: FixedChunkingConfig, *, tokenizer: Any | None = None) -> None:
        self.config = config
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(
            config.tokenizer.name,
            revision=config.tokenizer.revision,
            use_fast=True,
        )
        if not getattr(self.tokenizer, "is_fast", False):
            raise ValueError(
                "Fixed-token chunking requires a Hugging Face fast tokenizer: "
                "deterministic character provenance requires offset_mapping."
            )
        self.resolved_tokenizer_revision = _resolve_tokenizer_revision(self.tokenizer, config.tokenizer)
        self.chunker_config_hash = _chunker_config_hash(config, self.resolved_tokenizer_revision)

    @property
    def provenance(self) -> dict[str, str | int | None]:
        """Resolved chunker configuration suitable for experiment provenance."""
        return {
            "strategy": self.config.strategy,
            "tokenizer_name": self.config.tokenizer.name,
            "resolved_tokenizer_revision": self.resolved_tokenizer_revision,
            "max_tokens": self.config.max_tokens,
            "overlap_tokens": self.config.overlap_tokens,
            "chunker_config_hash": self.chunker_config_hash,
            "window_algorithm": "retokenized-v2",
        }

    @classmethod
    def from_resolved_config(cls, config: Mapping[str, Any], *, tokenizer: Any | None = None) -> "FixedTokenChunker":
        return cls(FixedChunkingConfig.from_resolved_config(config), tokenizer=tokenizer)

    def token_windows(self, document: Document) -> list[TokenWindow]:
        """Return exact, non-redundant windows for the complete canonical text."""
        validate_chunking_document(document)
        encoded = self.tokenizer(
            document.canonical_text,
            add_special_tokens=False,
            truncation=False,
            return_offsets_mapping=True,
        )
        token_ids = encoded["input_ids"]
        offsets = encoded["offset_mapping"]
        if len(token_ids) != len(offsets):
            raise ValueError("Tokenizer returned input_ids and offset_mapping of different lengths")
        _validate_offset_mapping(document.canonical_text, offsets)

        token_count = len(token_ids)
        if token_count == 0:
            return []

        windows: list[TokenWindow] = []
        token_start = 0
        while True:
            token_end = min(token_start + self.config.max_tokens, token_count)
            char_start = offsets[token_start][0]
            while token_end > token_start:
                char_end = offsets[token_end - 1][1]
                if len(
                    self.tokenizer(
                        document.canonical_text[char_start:char_end],
                        add_special_tokens=False,
                        truncation=False,
                        return_offsets_mapping=True,
                    )["input_ids"]
                ) <= self.config.max_tokens:
                    break
                token_end -= 1
            if token_end == token_start:
                raise ValueError("Tokenizer cannot fit one source token into the chunk budget")
            windows.append(TokenWindow(token_start, token_end, char_start, char_end))
            if token_end == token_count:
                return windows
            token_start = max(token_start + 1, token_end - self.config.overlap_tokens)

    def chunk(self, document: Document) -> list[Chunk]:
        """Create persisted chunk contracts from fixed token windows."""
        canonical_text = document.canonical_text
        text_hash = canonical_text_hash(document)
        block_starts = [block.char_start for block in document.blocks]
        chunks: list[Chunk] = []
        for window in self.token_windows(document):
            chunk_text = canonical_text[window.char_start : window.char_end]
            block_ids, pages, block_types, section_path = _provenance_for_span(
                document.blocks, block_starts, window.char_start, window.char_end
            )
            if not block_ids:
                raise ValueError("A non-empty token window must intersect at least one source block")
            chunks.append(
                Chunk(
                    chunk_id=_chunk_id(
                        doc_id=document.doc_id,
                        text_hash=text_hash,
                        config_hash=self.chunker_config_hash,
                        char_start=window.char_start,
                        char_end=window.char_end,
                    ),
                    doc_id=document.doc_id,
                    text=chunk_text,
                    block_ids=block_ids,
                    pages=pages,
                    block_types=block_types,
                    section_path=section_path,
                    char_start=window.char_start,
                    char_end=window.char_end,
                    token_count=len(
                        self.tokenizer(
                            chunk_text,
                            add_special_tokens=False,
                            truncation=False,
                            return_offsets_mapping=True,
                        )["input_ids"]
                    ),
                    oversize=False,
                    fallback_split=False,
                )
            )
        return chunks
