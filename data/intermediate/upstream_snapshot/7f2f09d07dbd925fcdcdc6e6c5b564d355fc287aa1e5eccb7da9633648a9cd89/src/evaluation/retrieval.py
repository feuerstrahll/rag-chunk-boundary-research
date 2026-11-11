"""Document, evidence, integrity, and paired uncertainty measurements."""

from __future__ import annotations

import math
import random
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from statistics import fmean, median
from typing import Sequence

from models import Chunk, Document
from ragstudy.data.watsonx_qa import GoldEvidence
from retrieval.ranking import RankedChunk, rank_documents


@dataclass(frozen=True)
class AlignedEvidence:
    evidence: GoldEvidence
    spans: tuple[tuple[int, int], ...]
    method: str = "unaligned"
    token_match_fraction: float = 0.0


def _normalized_with_offsets(text: str) -> tuple[str, list[int], list[int]]:
    normalized: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    for index, character in enumerate(text):
        for value in unicodedata.normalize("NFKC", character).casefold():
            if value.isspace():
                if normalized and normalized[-1] != " ":
                    normalized.append(" ")
                    starts.append(index)
                    ends.append(index + 1)
                elif normalized and normalized[-1] == " ":
                    ends[-1] = index + 1
            else:
                normalized.append(value)
                starts.append(index)
                ends.append(index + 1)
    if normalized and normalized[-1] == " ":
        normalized.pop()
        starts.pop()
        ends.pop()
    return "".join(normalized), starts, ends


def align_evidence(evidence: GoldEvidence, document: Document) -> AlignedEvidence:
    if evidence.doc_id != document.doc_id:
        raise ValueError("Evidence must be aligned only against its gold document")
    source, starts, ends = _normalized_with_offsets(document.canonical_text)
    target, _, _ = _normalized_with_offsets(evidence.text)
    spans: list[tuple[int, int]] = []
    if target:
        cursor = 0
        while (match := source.find(target, cursor)) >= 0:
            spans.append((starts[match], ends[match + len(target) - 1]))
            cursor = match + 1
    if spans:
        return AlignedEvidence(evidence, tuple(spans), "exact", 1.0)

    # The dataset's gold prose is plain text, while Markdown contains escapes,
    # comments, and link syntax. Align high-agreement word sequences without
    # treating an arbitrary short lexical overlap as a gold span.
    source_matches = list(re.finditer(r"\w+", source))
    gold_matches = list(re.finditer(r"\w+", target))
    source_words = [match.group() for match in source_matches]
    gold_words = [match.group() for match in gold_matches]
    if len(gold_words) < 5 or len(source_words) < 5:
        return AlignedEvidence(evidence, ())
    width = 5
    source_grams: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for index in range(len(source_words) - width + 1):
        source_grams[tuple(source_words[index : index + width])].append(index)
    anchors = set(range(0, len(gold_words) - width + 1, max(1, len(gold_words) // 20)))
    anchors.add(len(gold_words) - width)
    candidates: set[tuple[int, int]] = set()
    for gold_index in anchors:
        gram = tuple(gold_words[gold_index : gold_index + width])
        for source_index in source_grams.get(gram, ()):
            slack = max(10, len(gold_words) // 5)
            left = max(0, source_index - gold_index - slack)
            right = min(len(source_words), source_index + len(gold_words) - gold_index + slack)
            candidates.add((left, right))
    best = 0.0
    best_spans: set[tuple[int, int]] = set()
    for left, right in sorted(candidates):
        matching = SequenceMatcher(None, gold_words, source_words[left:right], autojunk=False).get_matching_blocks()
        matched = sum(block.size for block in matching)
        quality = matched / len(gold_words)
        if quality < 0.9:
            continue
        nonempty = [block for block in matching if block.size]
        first = left + nonempty[0].b
        last = left + nonempty[-1].b + nonempty[-1].size - 1
        span = (starts[source_matches[first].start()], ends[source_matches[last].end() - 1])
        if quality > best + 1e-9:
            best, best_spans = quality, {span}
        elif abs(quality - best) <= 1e-9:
            best_spans.add(span)
    if best_spans:
        return AlignedEvidence(evidence, tuple(sorted(best_spans)), "token_alignment", best)
    return AlignedEvidence(evidence, ())


def _coverage(span: tuple[int, int], chunks: Sequence[Chunk]) -> float:
    start, end = span
    if end <= start:
        return 0.0
    intervals = sorted((max(start, chunk.char_start), min(end, chunk.char_end)) for chunk in chunks)
    covered = 0
    right = start
    for left, stop in intervals:
        if stop > max(left, right):
            covered += stop - max(left, right)
            right = stop
    return covered / (end - start)


def evidence_scores(
    aligned: AlignedEvidence,
    ranking: Sequence[RankedChunk],
    all_gold_chunks: Sequence[Chunk],
    *,
    top_k: Sequence[int] = (1, 3, 5, 10),
    hit_threshold: float = 0.8,
) -> dict[str, float]:
    if not aligned.spans:
        raise ValueError("Cannot score unaligned evidence")
    if not 0 < hit_threshold <= 1:
        raise ValueError("Evidence hit threshold must be in (0, 1]")
    doc_id = aligned.evidence.doc_id
    gold_ranking = [item.chunk for item in ranking if item.chunk.doc_id == doc_id]
    result: dict[str, float] = {}
    for k in top_k:
        global_chunks = [item.chunk for item in ranking[:k] if item.chunk.doc_id == doc_id]
        coverage = max(_coverage(span, global_chunks) for span in aligned.spans)
        result[f"EvidenceCoverage@{k}"] = coverage
        result[f"EvidenceHit@{k}"] = float(coverage >= hit_threshold)
        result[f"GoldDocConditionalCoverage@{k}"] = max(
            _coverage(span, gold_ranking[:k]) for span in aligned.spans
        )
    result["MaxSingleChunkCoverage"] = max(
        (_coverage(span, [chunk]) for span in aligned.spans for chunk in all_gold_chunks), default=0.0
    )
    return result


def document_scores(ranking: Sequence[RankedChunk], gold_doc_ids: Sequence[str], *, top_k=(1, 3, 5, 10)) -> dict[str, float]:
    documents = rank_documents(ranking)
    gold = set(gold_doc_ids)
    if not gold:
        raise ValueError("Document metrics require at least one gold document")
    result: dict[str, float] = {}
    first = next((index for index, doc_id in enumerate(documents, start=1) if doc_id in gold), None)
    result["MRR"] = 1 / first if first else 0.0
    for k in top_k:
        result[f"Recall@{k}"] = len(set(documents[:k]) & gold) / len(gold)
        ideal = sum(1 / math.log2(index + 1) for index in range(1, min(k, len(gold)) + 1))
        gain = sum(1 / math.log2(index + 1) for index, doc_id in enumerate(documents[:k], 1) if doc_id in gold)
        result[f"nDCG@{k}"] = gain / ideal
    return result


def mean_question_scores(rows: dict[str, list[dict[str, float]]]) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """First average evidence within a question, then average questions equally."""
    per_question = {
        query_id: {key: fmean(item[key] for item in items) for key in items[0]}
        for query_id, items in rows.items()
        if items
    }
    if not per_question:
        return {}, {}
    means = {key: fmean(item[key] for item in per_question.values()) for key in next(iter(per_question.values()))}
    return means, per_question


def paired_bootstrap(
    fixed: dict[str, float], structure: dict[str, float], *, repetitions: int = 10_000, seed: int = 2026
) -> dict[str, float | int]:
    ids = sorted(set(fixed) & set(structure))
    if not ids:
        raise ValueError("Paired bootstrap has no common questions")
    differences = [structure[query_id] - fixed[query_id] for query_id in ids]
    rng = random.Random(seed)
    sampled = sorted(
        fmean(differences[rng.randrange(len(ids))] for _ in ids) for _ in range(repetitions)
    )
    return {
        "difference": fmean(differences),
        "ci_lower": sampled[int(0.025 * repetitions)],
        "ci_upper": sampled[min(repetitions - 1, int(0.975 * repetitions))],
        "question_count": len(ids),
    }


def integrity_statistics(documents: Sequence[Document], chunks: Sequence[Chunk]) -> dict[str, float | int]:
    by_doc: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        by_doc[chunk.doc_id].append(chunk)
    covered_chars = total_chars = split_blocks = block_count = 0
    crossing = fallback = 0
    duplicate_chars = emitted_chars = 0
    counts: list[int] = []
    for document in documents:
        current = by_doc[document.doc_id]
        counts.append(len(current))
        for block in document.blocks:
            if not block.text:
                continue
            block_count += 1
            total_chars += len(block.text)
            covered_chars += round(_coverage((block.char_start, block.char_end), current) * len(block.text))
            split_blocks += not any(chunk.char_start <= block.char_start and chunk.char_end >= block.char_end for chunk in current)
        for chunk in current:
            fallback += chunk.fallback_split
            contributing = [
                block.section_path
                for block in document.blocks
                if block.char_start < chunk.char_end and block.char_end > chunk.char_start
            ]
            crossing += len(set(contributing)) > 1
            emitted_chars += len(chunk.text)
        if current:
            start, end = current[0].char_start, current[0].char_end
            union = 0
            for chunk in sorted(current[1:], key=lambda item: (item.char_start, item.char_end)):
                if chunk.char_start > end:
                    union += end - start
                    start, end = chunk.char_start, chunk.char_end
                else:
                    end = max(end, chunk.char_end)
            union += end - start
            duplicate_chars += sum(len(chunk.text) for chunk in current) - union
    return {
        "chunk_count": len(chunks),
        "chunks_per_document_mean": fmean(counts) if counts else 0.0,
        "chunks_per_document_median": median(counts) if counts else 0.0,
        "block_char_coverage": covered_chars / total_chars if total_chars else 0.0,
        "split_block_fraction": split_blocks / block_count if block_count else 0.0,
        "cross_section_chunk_fraction": crossing / len(chunks) if chunks else 0.0,
        "fallback_split_fraction": fallback / len(chunks) if chunks else 0.0,
        "overlap_redundancy_fraction": duplicate_chars / emitted_chars if emitted_chars else 0.0,
    }
