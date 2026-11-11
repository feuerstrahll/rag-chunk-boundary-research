"""Explicit mapping from supported file extensions to parsers."""

from __future__ import annotations

from pathlib import Path

from .base import DocumentParser
from .markdown import MarkdownParser
from .plain_html import PlainHtmlParser
from .plain_pdf import PlainPdfParser
from .yaml_json import YamlJsonParser


class ParserRegistry:
    def __init__(self) -> None:
        structured = YamlJsonParser()
        self._by_suffix: dict[str, DocumentParser] = {
            ".pdf": PlainPdfParser(),
            ".html": PlainHtmlParser(),
            ".htm": PlainHtmlParser(),
            ".md": MarkdownParser(),
            ".markdown": MarkdownParser(),
            ".yaml": structured,
            ".yml": structured,
            ".json": structured,
        }

    def for_path(self, source: Path) -> DocumentParser:
        try:
            return self._by_suffix[source.suffix.lower()]
        except KeyError as error:
            raise ValueError(f"No baseline parser registered for {source.suffix or 'an extensionless file'}") from error

