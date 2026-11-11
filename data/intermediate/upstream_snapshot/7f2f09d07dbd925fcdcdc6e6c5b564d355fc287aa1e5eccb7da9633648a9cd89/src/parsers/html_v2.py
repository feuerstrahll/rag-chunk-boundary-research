"""Versioned HTML parser with approved non-nested readable containers."""

from __future__ import annotations

from pathlib import Path

from bs4 import BeautifulSoup, Tag

from models import Block, Document, EvidenceAnchor

from .base import DocumentParser

HTML_PARSER_VERSION = "html_parser_v2"


class HtmlParserV2(DocumentParser):
    """Keep v1 blocks and add approved containers without nested duplicates."""

    _REMOVE_TAGS = ("nav", "script", "style", "footer")
    _BASE_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6", "p", "li"}
    _APPROVED_TAGS = {"td", "th", "div", "span", "pre"}
    _STRUCTURAL_TAGS = _BASE_TAGS | {"td", "th", "pre"}

    def parse(self, source: Path, *, document_id: str, source_url: str | None = None) -> Document:
        return self.parse_text(
            source.read_text(encoding="utf-8"),
            document_id=document_id,
            source_url=source_url or source.as_uri(),
        )

    def parse_text(self, text: str, *, document_id: str, source_url: str) -> Document:
        soup = BeautifulSoup(text, "html.parser")
        for tag in soup.find_all(self._REMOVE_TAGS):
            tag.decompose()

        selected = self._select_non_nested(soup)
        selected_ids = {id(element) for element in selected}
        blocks: list[Block] = []
        section_path: list[str] = []
        heading_levels: list[int] = []
        all_elements = {id(element): index for index, element in enumerate(soup.find_all(True), start=1)}
        for block_index, element in enumerate(selected, start=1):
            value = element.get_text(" ", strip=True)
            if not value:
                continue
            tag_name = element.name.lower()
            level = int(tag_name[1]) if tag_name.startswith("h") else None
            if level is not None:
                while heading_levels and heading_levels[-1] >= level:
                    heading_levels.pop()
                    section_path.pop()
                heading_levels.append(level)
                section_path.append(value)
            block_type = self._block_type(tag_name)
            if tag_name == "li":
                value = f"- {value}"
            dom_index = all_elements[id(element)]
            element_id = element.get("id") or f"{tag_name}-{dom_index}"
            blocks.append(
                Block(
                    block_id=f"{document_id}:html-v2:{block_index}",
                    type=block_type,
                    text=value,
                    level=level,
                    section_path=tuple(section_path),
                    source_anchor=EvidenceAnchor(
                        format="html",
                        url=source_url,
                        element_id=element_id,
                        evidence_quote=value[:300],
                    ),
                    metadata={
                        "html_tag": tag_name,
                        "parser_version": HTML_PARSER_VERSION,
                        "dom_index": dom_index,
                        "dom_ancestry": [
                            all_elements[id(parent)] for parent in element.parents if isinstance(parent, Tag) and id(parent) in all_elements
                        ],
                        "selected_non_nested": id(element) in selected_ids,
                    },
                )
            )
        return Document(doc_id=document_id, source=source_url, blocks=blocks)

    def _select_non_nested(self, soup: BeautifulSoup) -> list[Tag]:
        candidates = [tag for tag in soup.find_all(self._BASE_TAGS | self._APPROVED_TAGS) if tag.get_text(" ", strip=True)]
        selected: list[Tag] = []
        for element in candidates:
            if any(parent in selected for parent in element.parents if isinstance(parent, Tag)):
                continue
            name = element.name.lower()
            if name in {"div", "span"} and element.find(self._STRUCTURAL_TAGS):
                continue
            selected.append(element)
        return selected

    @staticmethod
    def _block_type(tag_name: str) -> str:
        if tag_name.startswith("h"):
            return "heading"
        if tag_name == "li":
            return "list"
        if tag_name in {"td", "th"}:
            return "table"
        if tag_name == "pre":
            return "code"
        return "paragraph"
