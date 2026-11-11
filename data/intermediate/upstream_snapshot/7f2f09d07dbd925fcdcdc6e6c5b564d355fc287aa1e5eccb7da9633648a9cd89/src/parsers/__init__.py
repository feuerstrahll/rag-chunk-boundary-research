"""Baseline parsers that return only the project's unified document model."""

from .base import DocumentParser
from .html_v2 import HTML_PARSER_VERSION, HtmlParserV2
from .markdown import MarkdownParser
from .plain_html import PlainHtmlParser
from .plain_pdf import PlainPdfParser
from .registry import ParserRegistry
from .unstructured_parser import UnstructuredParser
from .yaml_json import YamlJsonParser

__all__ = [
    "DocumentParser",
    "HTML_PARSER_VERSION",
    "HtmlParserV2",
    "MarkdownParser",
    "ParserRegistry",
    "PlainHtmlParser",
    "PlainPdfParser",
    "UnstructuredParser",
    "YamlJsonParser",
]
