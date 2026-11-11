"""Project-owned contracts and canonical document representation."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

BlockType = Literal["heading", "paragraph", "list", "table", "code", "other"]
AnchorFormat = Literal["pdf", "html", "markdown", "yaml", "openapi"]


class EvidenceAnchor(BaseModel):
    """A verifiable location in the original source format."""

    model_config = ConfigDict(extra="forbid")
    format: AnchorFormat
    evidence_quote: str | None = None
    page: int | None = Field(default=None, ge=1)
    url: str | None = None
    element_id: str | None = None
    heading: str | None = None
    file: str | None = None
    line_start: int | None = Field(default=None, ge=1)
    line_end: int | None = Field(default=None, ge=1)
    yaml_path: str | None = None
    path: str | None = None
    http_method: str | None = None

    @model_validator(mode="after")
    def require_format_specific_fields(self) -> "EvidenceAnchor":
        if self.format == "pdf" and (self.page is None or not self.evidence_quote):
            raise ValueError("PDF anchors require page and evidence_quote")
        if self.format == "html" and (
            not self.url or not self.evidence_quote or not (self.element_id or self.heading)
        ):
            raise ValueError("HTML anchors require URL, element_id or heading, and evidence_quote")
        if self.format == "markdown" and (
            not self.file
            or not self.heading
            or self.line_start is None
            or self.line_end is None
            or self.line_end < self.line_start
        ):
            raise ValueError("Markdown anchors require file, heading, and an ordered line range")
        if self.format == "yaml" and not self.yaml_path:
            raise ValueError("YAML anchors require yaml_path")
        if self.format == "openapi" and (not self.path or not self.http_method):
            raise ValueError("OpenAPI anchors require path and HTTP method")
        return self


class ManifestEntry(BaseModel):
    """Immutable-source manifest record created by a collector."""

    model_config = ConfigDict(extra="forbid")
    document_id: str = Field(min_length=1)
    source_name: str = Field(min_length=1)
    source_url: str
    local_path: str = Field(min_length=1)
    format: str = Field(min_length=1)
    technology: str | None = None
    version: str | None = None
    document_type: str | None = None
    retrieved_at: datetime
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    collector: str = Field(min_length=1)


class Block(BaseModel):
    """A normalized text unit with deterministic offsets in its document."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    block_id: str = Field(min_length=1)
    type: BlockType = Field(validation_alias=AliasChoices("type", "block_type"))
    text: str
    level: int | None = Field(default=None, ge=1)
    page: int | None = Field(default=None, ge=1)
    section_path: tuple[str, ...] = ()
    bbox: tuple[float, float, float, float] | None = None
    char_start: int = Field(default=0, ge=0)
    char_end: int = Field(default=0, ge=0)
    source_anchor: EvidenceAnchor | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("text")
    @classmethod
    def forbid_text_changes_after_offsets(cls, value: str, info: ValidationInfo) -> str:
        """Normalization must finish before a canonical character range exists."""
        start = info.data.get("char_start", 0)
        end = info.data.get("char_end", 0)
        if end > start:
            raise ValueError("Block text is immutable after canonical offsets are assigned")
        return value

    @model_validator(mode="after")
    def validate_character_range(self) -> "Block":
        if self.char_end < self.char_start:
            raise ValueError("char_end must not precede char_start")
        return self

    @property
    def block_type(self) -> BlockType:
        """Compatibility accessor; all new code must use ``type``."""
        return self.type

    @property
    def page_numbers(self) -> list[int]:
        """Compatibility accessor for the previous multi-page placeholder."""
        return [] if self.page is None else [self.page]


class Document(BaseModel):
    """Normalized blocks plus their single derived canonical text representation."""

    model_config = ConfigDict(extra="forbid")
    doc_id: str = Field(min_length=1, validation_alias=AliasChoices("doc_id", "document_id"))
    source: str
    blocks: list[Block] = Field(default_factory=list)

    @model_validator(mode="after")
    def assign_offsets_on_construction(self) -> "Document":
        from normalization.canonical import assign_block_offsets

        assign_block_offsets(self)
        return self

    @property
    def canonical_text(self) -> str:
        """Derived from normalized blocks; no duplicate text copy is stored."""
        from normalization.canonical import build_canonical_text

        return build_canonical_text(self)

    @property
    def document_id(self) -> str:
        """Compatibility accessor; all new code must use ``doc_id``."""
        return self.doc_id


class ChunkMetadata(BaseModel):
    """Source information derived from the document and contributing blocks."""

    model_config = ConfigDict(extra="forbid")
    title: str | None = None
    source_url: str | None = None
    representation: str | None = None
    source_format: str | None = None
    source_anchors: tuple[EvidenceAnchor, ...] = ()


class Chunk(BaseModel):
    """Chunk provenance contract for future fixed and structure-aware chunkers."""

    model_config = ConfigDict(extra="forbid")
    chunk_id: str = Field(min_length=1)
    doc_id: str = Field(min_length=1, validation_alias=AliasChoices("doc_id", "document_id"))
    text: str
    block_ids: tuple[str, ...] = Field(min_length=1)
    pages: tuple[int, ...] = ()
    section_path: tuple[str, ...] = ()
    block_types: tuple[str, ...] = ()
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    token_count: int = Field(ge=0)
    oversize: bool = False
    fallback_split: bool = False
    metadata: ChunkMetadata | None = None

    @model_validator(mode="after")
    def validate_character_range(self) -> "Chunk":
        if self.char_end < self.char_start:
            raise ValueError("char_end must not precede char_start")
        return self

    @property
    def document_id(self) -> str:
        """Compatibility accessor; all new code must use ``doc_id``."""
        return self.doc_id
