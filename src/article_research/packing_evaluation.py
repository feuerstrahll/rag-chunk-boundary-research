"""Canonical-character evidence metrics and paired document-cluster inference.

Evidence definitions preserve upstream article_research.step5b.question_metrics;
context packing preserves article_research.stage4.select_ranked_context.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

METRICS = (
    "EvidenceCoverage@B=1024", "EvidenceCoverage@5", "EvidenceRecall@5",
    "EvidenceMRR@5", "DocumentRecall@5", "DocumentMRR@10",
)


def select_ranked_context(ranking: Sequence[Any], budget: int) -> tuple[list[Any], int]:
    selected, used = [], 0
    for item in ranking:
        if item.chunk.token_count <= budget - used:
            selected.append(item)
            used += item.chunk.token_count
    return selected, used


def coverage(span: tuple[int, int], chunks: Sequence[Any]) -> float:
    start, end = span
    right, covered = start, 0
    for left, stop in sorted((max(start, c.char_start), min(end, c.char_end)) for c in chunks):
        if stop > max(left, right):
            covered += stop - max(left, right)
            right = stop
    return covered / (end - start)


def multi_span_coverage(spans: Sequence[tuple[int, int]], chunks: Sequence[Any]) -> float:
    return sum((b - a) * coverage((a, b), chunks) for a, b in spans) / sum(b - a for a, b in spans)


def evidence_metrics(ranking: Sequence[Any], doc_id: str, spans: Sequence[tuple[int, int]]) -> dict[str, float | int]:
    top = [item.chunk for item in ranking[:5] if item.chunk.doc_id == doc_id]
    top_coverage = multi_span_coverage(spans, top)
    first = next((item.rank for item in ranking[:5] if item.chunk.doc_id == doc_id
                  and multi_span_coverage(spans, [item.chunk]) >= 0.8), None)
    selected, used = select_ranked_context(ranking, 1024)
    return {
        "EvidenceCoverage@5": top_coverage,
        "EvidenceRecall@5": float(top_coverage >= 0.8),
        "EvidenceMRR@5": 1 / first if first else 0.0,
        "EvidenceCoverage@B=1024": multi_span_coverage(
            spans, [item.chunk for item in selected if item.chunk.doc_id == doc_id]),
        "actual_context_tokens_B1024": used,
    }


def document_metrics(ranking: Sequence[Any], gold_doc_ids: Sequence[str]) -> dict[str, float]:
    """Max-pool every supplied candidate before cutting unique-document ranks."""
    scores: dict[str, float] = {}
    for item in ranking:
        doc_id = item.chunk.doc_id
        scores[doc_id] = max(scores.get(doc_id, -float("inf")), item.score)
    documents = sorted(scores, key=lambda doc: (-scores[doc], doc))
    gold = set(gold_doc_ids)
    if not gold:
        raise ValueError("Empty document gold")
    first = next((rank for rank, doc in enumerate(documents[:10], 1) if doc in gold), None)
    return {
        "DocumentRecall@5": len(set(documents[:5]) & gold) / len(gold),
        "DocumentMRR@10": 1 / first if first else 0.0,
    }


def document_clusters(question_documents: dict[str, Sequence[str]]) -> dict[str, str]:
    """Connected components of the question--gold-document bipartite graph."""
    parent: dict[str, str] = {}

    def find(doc: str) -> str:
        parent.setdefault(doc, doc)
        if parent[doc] != doc:
            parent[doc] = find(parent[doc])
        return parent[doc]

    for docs in question_documents.values():
        if not docs:
            raise ValueError("Every question needs gold documents")
        root = find(docs[0])
        for doc in docs[1:]:
            a, b = find(root), find(doc)
            parent[max(a, b)] = min(a, b)
    return {qid: find(docs[0]) for qid, docs in question_documents.items()}


def paired_cluster_bootstrap(
    reference: dict[str, float], candidate: dict[str, float], clusters: dict[str, str],
    *, repetitions: int = 10000, seed: int = 2026,
) -> dict[str, Any]:
    if reference.keys() != candidate.keys() or reference.keys() != clusters.keys() or not reference:
        raise ValueError("Bootstrap requires identical nonempty paired IDs and cluster membership")
    keys = sorted(set(clusters.values()))
    differences = {qid: candidate[qid] - reference[qid] for qid in reference}
    sums = np.array([sum(differences[q] for q in sorted(differences) if clusters[q] == c) for c in keys])
    sizes = np.array([sum(value == c for value in clusters.values()) for c in keys])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(keys), size=(repetitions, len(keys)))
    sampled = sums[draws].sum(axis=1) / sizes[draws].sum(axis=1)
    low, high = np.quantile(sampled, [0.025, 0.975])
    return {
        "difference": float(np.mean(list(differences.values()))),
        "ci_lower": float(low), "ci_upper": float(high),
        "cluster_count": len(keys), "question_count": len(reference),
        "repetitions": repetitions, "seed": seed,
        "bootstrap_method": "paired_document_connected_component_cluster_bootstrap",
        "interpretation": "exploratory_descriptive_unadjusted_for_multiplicity",
    }
