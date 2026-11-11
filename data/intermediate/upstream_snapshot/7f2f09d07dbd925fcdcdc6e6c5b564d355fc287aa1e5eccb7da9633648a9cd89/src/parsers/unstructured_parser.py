"""Optional Unstructured adapter; no Unstructured object leaves this module."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from models import Block, Document, EvidenceAnchor

from .base import DocumentParser


class UnstructuredParser(DocumentParser):
    """Translate Unstructured elements into project-owned blocks."""

    _TYPE_MAP = {
        "Title": "heading",
        "NarrativeText": "paragraph",
        "Table": "table",
        "ListItem": "list",
    }

    def parse(self, source: Path, *, document_id: str, source_url: str | None = None) -> Document:
        try:
            from unstructured.partition.auto import partition
        except ImportError as error:
            raise RuntimeError("Unstructured is required to use UnstructuredParser") from error
        elements = partition(filename=str(source))
        blocks = [self._to_block(element, document_id, index + 1) for index, element in enumerate(elements)]
        return Document(
            doc_id=document_id,
            source=source_url or str(source),
            blocks=blocks,
        )

    def _to_block(self, element: Any, document_id: str, index: int) -> Block:
        """Make a project block immediately; ``element`` is never returned or stored."""
        text = str(getattr(element, "text", "")).strip()
        element_type = type(element).__name__
        metadata = getattr(element, "metadata", None)
        page_number = getattr(metadata, "page_number", None)
        pages = [page_number] if isinstance(page_number, int) and page_number > 0 else []
        anchor = (
            EvidenceAnchor(format="pdf", page=page_number, evidence_quote=text[:300])
            if pages and text
            else None
        )
        return Block(
            block_id=f"{document_id}:unstructured:{index}",
            type=self._TYPE_MAP.get(element_type, "other"),
            text=text,
            page=page_number if pages else None,
            source_anchor=anchor,
            metadata={"unstructured_category": element_type},
        )
