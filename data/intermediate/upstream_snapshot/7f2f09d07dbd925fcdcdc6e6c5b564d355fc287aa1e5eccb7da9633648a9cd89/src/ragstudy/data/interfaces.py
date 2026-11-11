"""Stable interfaces separating datasets from pipeline stages."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class DocumentSource(BaseModel):
    document_id: str = Field(min_length=1)
    content: str | bytes
    source_format: str = Field(min_length=1)
    provenance: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Query(BaseModel):
    query_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class GoldLabels(BaseModel):
    """Dataset-independent query-to-document and evidence references."""

    document_ids_by_query: dict[str, list[str]] = Field(default_factory=dict)
    evidence_by_query: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)


class LayoutGold(BaseModel):
    records: list[dict[str, Any]] = Field(default_factory=list)


class RetrievalDatasetLoader(ABC):
    @abstractmethod
    def load_documents(self) -> list[DocumentSource]: ...

    @abstractmethod
    def load_queries(self) -> list[Query]: ...

    @abstractmethod
    def load_gold(self) -> GoldLabels: ...


class ParsingDatasetLoader(ABC):
    @abstractmethod
    def load_documents(self) -> list[DocumentSource]: ...

    @abstractmethod
    def load_layout_gold(self) -> LayoutGold: ...
