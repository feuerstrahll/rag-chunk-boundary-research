"""YAML and JSON parser that retains paths instead of flattening structure away."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from models import Block, Document, EvidenceAnchor

from .base import DocumentParser


class YamlJsonParser(DocumentParser):
    def parse(self, source: Path, *, document_id: str, source_url: str | None = None) -> Document:
        raw = source.read_text(encoding="utf-8")
        format_name = "json" if source.suffix.lower() == ".json" else "yaml"
        parsed = json.loads(raw) if format_name == "json" else yaml.safe_load(raw)
        blocks: list[Block] = []
        self._walk(parsed, (), document_id, blocks)
        return Document(
            doc_id=document_id,
            source=source_url or str(source),
            blocks=blocks,
        )

    def _walk(self, value: Any, path: tuple[str, ...], document_id: str, blocks: list[Block]) -> None:
        if isinstance(value, dict):
            if not value:
                self._append_leaf(value, path, document_id, blocks)
            for key, child in value.items():
                self._walk(child, (*path, str(key)), document_id, blocks)
        elif isinstance(value, list):
            if not value:
                self._append_leaf(value, path, document_id, blocks)
            for index, child in enumerate(value):
                self._walk(child, (*path, str(index)), document_id, blocks)
        else:
            self._append_leaf(value, path, document_id, blocks)

    @staticmethod
    def _append_leaf(value: Any, path: tuple[str, ...], document_id: str, blocks: list[Block]) -> None:
        structural_path = ".".join(path) or "$"
        rendered = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
        blocks.append(
            Block(
                block_id=f"{document_id}:config:{len(blocks) + 1}",
                type="other",
                text=f"{structural_path}: {rendered}",
                section_path=list(path[:-1]),
                source_anchor=EvidenceAnchor(format="yaml", yaml_path=structural_path),
                metadata={"structural_path": structural_path, "value_type": type(value).__name__},
            )
        )

