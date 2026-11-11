"""Load pinned watsonxDocsQA questions with explicit evidence-to-document pairs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


@dataclass(frozen=True)
class GoldEvidence:
    doc_id: str
    text: str


@dataclass(frozen=True)
class QARecord:
    query_id: str
    question: str
    answer: str
    split: str
    evidence: tuple[GoldEvidence, ...]

    @property
    def gold_doc_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(item.doc_id for item in self.evidence))


def _values(raw: Any, *, field: str) -> tuple[str, ...]:
    if isinstance(raw, str):
        if raw.lstrip().startswith("["):
            raw = json.loads(raw)
        else:
            raw = [raw]
    if not isinstance(raw, (list, tuple)) or not raw or any(not isinstance(value, str) or not value for value in raw):
        raise ValueError(f"Invalid QA field {field}: expected one or more non-empty strings")
    return tuple(raw)


def load_qa_parquets(paths_by_split: dict[str, Path], *, corpus_doc_ids: set[str]) -> list[QARecord]:
    records: list[QARecord] = []
    seen: set[str] = set()
    for split, path in sorted(paths_by_split.items()):
        for row in pq.read_table(path).to_pylist():
            query_id = row.get("question_id")
            if not isinstance(query_id, str) or not query_id or query_id in seen:
                raise ValueError(f"Duplicate or missing QA question_id: {query_id!r}")
            seen.add(query_id)
            ids = _values(
                row.get("ground_truths_contexts_ids", row.get("correct_answer_document_ids")),
                field="gold document IDs",
            )
            texts = _values(row.get("ground_truths_contexts"), field="gold contexts")
            if len(ids) != len(texts):
                raise ValueError(f"Gold IDs and contexts have different lengths for {query_id}")
            if set(ids) - corpus_doc_ids:
                raise ValueError(f"Gold document missing from corpus for {query_id}: {sorted(set(ids) - corpus_doc_ids)}")
            question, answer = row.get("question"), row.get("correct_answer")
            if not isinstance(question, str) or not question or not isinstance(answer, str) or not answer:
                raise ValueError(f"Missing question or answer for {query_id}")
            records.append(
                QARecord(
                    query_id=query_id,
                    question=question,
                    answer=answer,
                    split=split,
                    evidence=tuple(GoldEvidence(doc_id, text) for doc_id, text in zip(ids, texts)),
                )
            )
    return sorted(records, key=lambda record: record.query_id)
