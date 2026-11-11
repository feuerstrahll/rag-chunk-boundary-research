"""Section-boundary chunking over contiguous canonical document spans."""

from __future__ import annotations

from hashlib import sha256
from typing import Any, Mapping

import orjson
from pydantic import BaseModel, ConfigDict, model_validator
from transformers import AutoTokenizer

from models import Block, Chunk, Document

from .fixed import (
    TokenizerConfig,
    _chunk_id,
    _provenance_for_span,
    _resolve_tokenizer_revision,
    _validate_offset_mapping,
    canonical_text_hash,
    validate_chunking_document,
)


class StructureChunkingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: str = "structure"
    tokenizer: TokenizerConfig
    max_tokens: int
    overlap_tokens: int

    @model_validator(mode="after")
    def validate_limits(self) -> "StructureChunkingConfig":
        if self.strategy != "structure" or self.max_tokens <= 0:
            raise ValueError("Structure chunking requires strategy='structure' and positive max_tokens")
        if not 0 <= self.overlap_tokens < self.max_tokens:
            raise ValueError("overlap_tokens must be between zero and max_tokens")
        return self


class StructureAwareChunker:
    """Pack whole blocks inside one section and split only oversized blocks."""

    def __init__(self, config: StructureChunkingConfig, *, tokenizer: Any | None = None) -> None:
        self.config = config
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(
            config.tokenizer.name, revision=config.tokenizer.revision, use_fast=True
        )
        if not getattr(self.tokenizer, "is_fast", False):
            raise ValueError("Structure-aware chunking requires a fast tokenizer with character offsets")
        self.resolved_tokenizer_revision = _resolve_tokenizer_revision(self.tokenizer, config.tokenizer)
        identity = {
            "strategy": config.strategy,
            "tokenizer_name": config.tokenizer.name,
            "resolved_tokenizer_revision": self.resolved_tokenizer_revision,
            "max_tokens": config.max_tokens,
            "overlap_tokens": config.overlap_tokens,
        }
        self.chunker_config_hash = sha256(orjson.dumps(identity, option=orjson.OPT_SORT_KEYS)).hexdigest()

    @property
    def provenance(self) -> dict[str, str | int | None]:
        return {
            "strategy": self.config.strategy,
            "tokenizer_name": self.config.tokenizer.name,
            "resolved_tokenizer_revision": self.resolved_tokenizer_revision,
            "max_tokens": self.config.max_tokens,
            "overlap_tokens": self.config.overlap_tokens,
            "chunker_config_hash": self.chunker_config_hash,
        }

    @classmethod
    def from_resolved_config(
        cls, config: Mapping[str, Any], *, tokenizer: Any | None = None
    ) -> "StructureAwareChunker":
        chunking = config.get("chunking")
        if not isinstance(chunking, Mapping):
            raise ValueError("Resolved configuration must contain a chunking mapping")
        return cls(StructureChunkingConfig.model_validate(chunking), tokenizer=tokenizer)

    def _count(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"])

    def _make_chunk(self, document: Document, start: int, end: int, *, fallback: bool) -> Chunk:
        text = document.canonical_text[start:end]
        block_ids, pages, block_types, section_path = _provenance_for_span(
            document.blocks, [block.char_start for block in document.blocks], start, end
        )
        if not text or not block_ids:
            raise ValueError("Chunk must contain text from at least one source block")
        count = self._count(text)
        if count > self.config.max_tokens:
            raise ValueError("Structure chunk exceeds its token budget")
        return Chunk(
            chunk_id=_chunk_id(
                doc_id=document.doc_id,
                text_hash=canonical_text_hash(document),
                config_hash=self.chunker_config_hash,
                char_start=start,
                char_end=end,
            ),
            doc_id=document.doc_id,
            text=text,
            block_ids=block_ids,
            pages=pages,
            block_types=block_types,
            section_path=section_path,
            char_start=start,
            char_end=end,
            token_count=count,
            fallback_split=fallback,
        )

    def _split_block(self, document: Document, block: Block) -> list[Chunk]:
        encoded = self.tokenizer(
            block.text, add_special_tokens=False, truncation=False, return_offsets_mapping=True
        )
        offsets = encoded["offset_mapping"]
        if len(encoded["input_ids"]) != len(offsets):
            raise ValueError("Tokenizer returned incompatible token and offset counts")
        _validate_offset_mapping(block.text, offsets)
        chunks: list[Chunk] = []
        position = 0
        while position < len(offsets):
            end = min(position + self.config.max_tokens, len(offsets))
            while end > position and self._count(block.text[offsets[position][0] : offsets[end - 1][1]]) > self.config.max_tokens:
                end -= 1
            if end == position:
                raise ValueError("Tokenizer cannot fit one source token into the fallback budget")
            chunks.append(
                self._make_chunk(
                    document,
                    block.char_start + offsets[position][0],
                    block.char_start + offsets[end - 1][1],
                    fallback=True,
                )
            )
            if end == len(offsets):
                break
            position = max(position + 1, end - self.config.overlap_tokens)
        return chunks

    def chunk(self, document: Document) -> list[Chunk]:
        validate_chunking_document(document)
        segments: list[list[Block]] = []
        for block in document.blocks:
            if not block.text:
                continue
            if not segments or block.type == "heading" or block.section_path != segments[-1][-1].section_path:
                segments.append([])
            segments[-1].append(block)

        chunks: list[Chunk] = []
        for segment in segments:
            active: list[Block] = []
            for block in segment:
                if self._count(block.text) > self.config.max_tokens:
                    if active:
                        chunks.append(self._make_chunk(document, active[0].char_start, active[-1].char_end, fallback=False))
                        active = []
                    chunks.extend(self._split_block(document, block))
                    continue
                if not active:
                    active = [block]
                    continue
                candidate = document.canonical_text[active[0].char_start : block.char_end]
                if self._count(candidate) <= self.config.max_tokens:
                    active.append(block)
                    continue
                chunks.append(self._make_chunk(document, active[0].char_start, active[-1].char_end, fallback=False))
                overlap: list[Block] = []
                for previous in reversed(active):
                    trial = [previous, *overlap]
                    overlap_text = document.canonical_text[trial[0].char_start : trial[-1].char_end]
                    if self._count(overlap_text) > self.config.overlap_tokens:
                        break
                    overlap = trial
                while overlap and self._count(document.canonical_text[overlap[0].char_start : block.char_end]) > self.config.max_tokens:
                    overlap.pop(0)
                active = [*overlap, block]
            if active:
                chunks.append(self._make_chunk(document, active[0].char_start, active[-1].char_end, fallback=False))
        return chunks
