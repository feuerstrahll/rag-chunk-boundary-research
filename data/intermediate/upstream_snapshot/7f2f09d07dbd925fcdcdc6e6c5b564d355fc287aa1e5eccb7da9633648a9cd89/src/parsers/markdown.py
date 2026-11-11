"""Line-oriented Markdown baseline with heading-aware anchors."""

from __future__ import annotations

import re
from pathlib import Path

from models import Block, Document, EvidenceAnchor

from .base import DocumentParser

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_LIST_ITEM = re.compile(r"^\s*(?:[-*+] |\d+[.)] )(.+?)\s*$")


class MarkdownParser(DocumentParser):
    def parse(self, source: Path, *, document_id: str, source_url: str | None = None) -> Document:
        return self.parse_text(
            source.read_text(encoding="utf-8"),
            document_id=document_id,
            source_label=str(source),
            source_url=source_url,
        )

    def parse_text(
        self, text: str, *, document_id: str, source_label: str, source_url: str | None = None
    ) -> Document:
        """Parse a dataset-provided Markdown representation without a temporary file."""
        lines = text.splitlines()
        blocks: list[Block] = []
        section_path: list[str] = []
        heading_levels: list[int] = []
        active_heading = "__root__"
        paragraph: list[str] = []
        paragraph_start = 1

        def append_paragraph(end_line: int) -> None:
            nonlocal paragraph
            text = " ".join(part.strip() for part in paragraph).strip()
            if text:
                blocks.append(
                    Block(
                        block_id=f"{document_id}:markdown:{len(blocks) + 1}",
                        block_type="paragraph",
                        text=text,
                        section_path=list(section_path),
                        source_anchor=EvidenceAnchor(
                            format="markdown",
                            file=source_label,
                            heading=active_heading,
                            line_start=paragraph_start,
                            line_end=end_line,
                        ),
                    )
                )
            paragraph = []

        for line_number, line in enumerate(lines, start=1):
            heading_match = _HEADING.match(line)
            list_match = _LIST_ITEM.match(line)
            if heading_match:
                append_paragraph(line_number - 1)
                level, text = len(heading_match.group(1)), heading_match.group(2)
                while heading_levels and heading_levels[-1] >= level:
                    heading_levels.pop()
                    section_path.pop()
                heading_levels.append(level)
                section_path.append(text)
                active_heading = text
                blocks.append(
                    Block(
                        block_id=f"{document_id}:markdown:{len(blocks) + 1}",
                        type="heading",
                        text=text,
                        level=level,
                        section_path=list(section_path),
                        source_anchor=EvidenceAnchor(
                            format="markdown", file=source_label, heading=text, line_start=line_number, line_end=line_number
                        ),
                    )
                )
            elif list_match:
                append_paragraph(line_number - 1)
                text = list_match.group(1)
                blocks.append(
                    Block(
                        block_id=f"{document_id}:markdown:{len(blocks) + 1}",
                        type="list",
                        text=f"- {text}",
                        section_path=list(section_path),
                        source_anchor=EvidenceAnchor(
                            format="markdown", file=source_label, heading=active_heading, line_start=line_number, line_end=line_number
                        ),
                    )
                )
            elif not line.strip():
                append_paragraph(line_number - 1)
            else:
                if not paragraph:
                    paragraph_start = line_number
                paragraph.append(line)
        append_paragraph(len(lines))
        return Document(
            doc_id=document_id,
            source=source_url or source_label,
            blocks=blocks,
        )

