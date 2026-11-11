"""Dataset-owned loader for audited watsonxDocsQA corpus representations."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pyarrow.parquet as pq
from datasets import Dataset

from .interfaces import DocumentSource

WatsonxRepresentation = Literal["document", "md_document", "html_document"]
WATSONX_REPRESENTATIONS: tuple[WatsonxRepresentation, ...] = ("document", "md_document", "html_document")


class WatsonxDocsQACorpusLoader:
    """Read one confirmed corpus representation from the original local parquet file."""

    def __init__(self, dataset: Dataset, *, revision: str) -> None:
        self.dataset = dataset
        self.revision = revision

    @classmethod
    def from_parquet(cls, path: Path, *, revision: str) -> "WatsonxDocsQACorpusLoader":
        dataset = Dataset(pq.read_table(path))
        required_fields = {"doc_id", "url", *WATSONX_REPRESENTATIONS}
        missing = required_fields - set(dataset.column_names)
        if missing:
            raise ValueError(f"watsonxDocsQA corpus is missing audited fields: {sorted(missing)}")
        return cls(dataset, revision=revision)

    @property
    def row_count(self) -> int:
        return len(self.dataset)

    @property
    def fields(self) -> tuple[str, ...]:
        return tuple(self.dataset.column_names)

    def documents(self, representation: WatsonxRepresentation) -> list[DocumentSource]:
        if representation not in WATSONX_REPRESENTATIONS:
            raise ValueError(f"Unsupported watsonxDocsQA representation: {representation}")
        records: list[DocumentSource] = []
        for row in self.dataset:
            doc_id, content, url = row["doc_id"], row[representation], row["url"]
            if not isinstance(doc_id, str) or not doc_id:
                raise ValueError("watsonxDocsQA contains an empty doc_id")
            if not isinstance(content, str):
                raise ValueError(f"watsonxDocsQA {representation} is not text for {doc_id!r}")
            records.append(
                DocumentSource(
                    document_id=doc_id,
                    content=content,
                    source_format={"document": "text", "md_document": "markdown", "html_document": "html"}[representation],
                    provenance={
                        "dataset": "ibm-research/watsonxDocsQA",
                        "revision": self.revision,
                        "representation": representation,
                        "source_url": url,
                    },
                    metadata={"title": row.get("title")},
                )
            )
        return sorted(records, key=lambda record: record.document_id)
