"""I0 PDF baseline: one plain-text block per page."""

from __future__ import annotations

from pathlib import Path

import fitz

from models import Block, Document, EvidenceAnchor

from .base import DocumentParser


class PlainPdfParser(DocumentParser):
    """Extract PDF text with PyMuPDF without structural inference."""

    def parse(self, source: Path, *, document_id: str, source_url: str | None = None) -> Document:
        blocks: list[Block] = []
        with fitz.open(source) as pdf:
            for page_number, page in enumerate(pdf, start=1):
                text = page.get_text("text").strip()
                if not text:
                    continue
                blocks.append(
                    Block(
                        block_id=f"{document_id}:page:{page_number}",
                        type="paragraph",
                        text=text,
                        page=page_number,
                        source_anchor=EvidenceAnchor(
                            format="pdf", page=page_number, evidence_quote=text[:300]
                        ),
                    )
                )
        return Document(
            doc_id=document_id,
            source=source_url or str(source),
            blocks=blocks,
        )

