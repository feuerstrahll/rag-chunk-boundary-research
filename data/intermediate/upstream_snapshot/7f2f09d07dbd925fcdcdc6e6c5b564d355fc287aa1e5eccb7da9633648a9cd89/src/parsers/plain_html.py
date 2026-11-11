"""Readable-text HTML baseline using BeautifulSoup."""

from __future__ import annotations

from pathlib import Path

from bs4 import BeautifulSoup

from models import Block, Document, EvidenceAnchor

from .base import DocumentParser


class PlainHtmlParser(DocumentParser):
    """Discard boilerplate elements while retaining readable body text."""

    _REMOVE_TAGS = ("nav", "script", "style", "footer")

    def parse(self, source: Path, *, document_id: str, source_url: str | None = None) -> Document:
        return self.parse_text(
            source.read_text(encoding="utf-8"),
            document_id=document_id,
            source_url=source_url or source.as_uri(),
        )

    def parse_text(self, text: str, *, document_id: str, source_url: str) -> Document:
        """Parse an in-memory HTML dataset representation with stable source provenance."""
        soup = BeautifulSoup(text, "html.parser")
        for tag in soup.find_all(self._REMOVE_TAGS):
            tag.decompose()

        blocks: list[Block] = []
        section_path: list[str] = []
        heading_levels: list[int] = []
        for index, element in enumerate(soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li"]), start=1):
            text = element.get_text(" ", strip=True)
            if not text:
                continue
            tag_name = element.name.lower()
            block_type = "heading" if tag_name.startswith("h") else "list" if tag_name == "li" else "paragraph"
            if tag_name == "li":
                text = f"- {text}"
            level = int(tag_name[1]) if tag_name.startswith("h") else None
            if level is not None:
                while heading_levels and heading_levels[-1] >= level:
                    heading_levels.pop()
                    section_path.pop()
                heading_levels.append(level)
                section_path.append(text)
            element_id = element.get("id") or f"{tag_name}-{index}"
            blocks.append(
                Block(
                    block_id=f"{document_id}:html:{index}",
                    type=block_type,
                    text=text,
                    level=level,
                    section_path=tuple(section_path),
                    source_anchor=EvidenceAnchor(
                        format="html",
                        url=source_url,
                        element_id=element_id,
                        evidence_quote=text[:300],
                    ),
                    metadata={"html_tag": tag_name},
                )
            )
        return Document(
            doc_id=document_id,
            source=source_url,
            blocks=blocks,
        )

