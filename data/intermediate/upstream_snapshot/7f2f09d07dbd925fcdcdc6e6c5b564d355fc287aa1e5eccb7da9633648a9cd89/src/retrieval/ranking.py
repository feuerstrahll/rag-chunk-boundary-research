"""Deterministic chunk ranking and document projection for small local corpora."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from models import Chunk

_WORDS = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True)
class RankedChunk:
    chunk: Chunk
    score: float
    rank: int


def _sort_scored(chunks: Sequence[Chunk], scores: dict[int, float]) -> list[RankedChunk]:
    ordered = sorted(
        scores,
        key=lambda index: (-scores[index], chunks[index].doc_id, chunks[index].chunk_id),
    )
    return [RankedChunk(chunks[index], float(scores[index]), rank) for rank, index in enumerate(ordered, start=1)]


class BM25Index:
    def __init__(self, chunks: Sequence[Chunk], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = list(chunks)
        self.k1, self.b = k1, b
        self.lengths: list[int] = []
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for index, chunk in enumerate(self.chunks):
            terms = _WORDS.findall(chunk.text.casefold())
            self.lengths.append(len(terms))
            for term, count in Counter(terms).items():
                self.postings[term].append((index, count))
        self.average_length = sum(self.lengths) / len(self.lengths) if self.lengths else 0.0

    def search(self, query: str) -> list[RankedChunk]:
        scores: dict[int, float] = defaultdict(float)
        size = len(self.chunks)
        if not size or not self.average_length:
            return []
        for term, query_count in Counter(_WORDS.findall(query.casefold())).items():
            postings = self.postings.get(term, ())
            if not postings:
                continue
            idf = math.log1p((size - len(postings) + 0.5) / (len(postings) + 0.5))
            for index, frequency in postings:
                length_scale = 1 - self.b + self.b * self.lengths[index] / self.average_length
                scores[index] += query_count * idf * frequency * (self.k1 + 1) / (frequency + self.k1 * length_scale)
        return _sort_scored(self.chunks, scores)


def resolve_text_budget(model: Any, *, requested_max_tokens: int) -> int:
    """Use the smallest real encoder limit after accounting for special tokens."""
    tokenizer = model.tokenizer
    limits = [getattr(model, "max_seq_length", None), getattr(tokenizer, "model_max_length", None)]
    finite = [int(limit) for limit in limits if isinstance(limit, int) and 0 < limit < 1_000_000]
    if not finite:
        raise ValueError("Dense encoder does not expose a finite sequence length")
    special = tokenizer.num_special_tokens_to_add(pair=False)
    budget = min(requested_max_tokens, min(finite) - special)
    if budget <= 0:
        raise ValueError("Dense encoder has no text-token capacity")
    return budget


def validate_encoder_inputs(chunks: Sequence[Chunk], model: Any) -> None:
    tokenizer = model.tokenizer
    limit = min(
        int(value)
        for value in (model.max_seq_length, tokenizer.model_max_length)
        if isinstance(value, int) and 0 < value < 1_000_000
    )
    for offset in range(0, len(chunks), 32):
        batch = chunks[offset : offset + 32]
        expected_ids = []
        for chunk in batch:
            encoded = tokenizer(chunk.text, add_special_tokens=True, truncation=False)
            ids = encoded["input_ids"]
            if len(ids) > limit:
                raise ValueError(f"Dense encoder would truncate chunk {chunk.chunk_id}")
            expected_ids.append(ids)
        # SentenceTransformer may apply another tokenization/truncation path.
        # Check the IDs that actually reach its encoder, without padding.
        if callable(getattr(model, "tokenize", None)):
            actual = model.tokenize([chunk.text for chunk in batch])
            if "input_ids" not in actual or "attention_mask" not in actual:
                raise ValueError("Dense encoder tokenization lacks input IDs or attention mask")
            for chunk, expected, ids, mask in zip(
                batch, expected_ids, actual["input_ids"], actual["attention_mask"]
            ):
                length = int(mask.sum())
                shown = ids[:length].tolist() if hasattr(ids, "tolist") else list(ids[:length])
                if shown != expected:
                    raise ValueError(f"Dense encoder additionally truncates or changes chunk {chunk.chunk_id}")


def encode_chunks(
    chunks: Sequence[Chunk], model: Any, *, batch_size: int = 32, validate: bool = True
) -> np.ndarray:
    """Build the normalized float32 matrix used by production dense retrieval."""
    if validate:
        validate_encoder_inputs(chunks, model)
    return np.asarray(
        model.encode(
            [chunk.text for chunk in chunks],
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ),
        dtype=np.float32,
    )


class DenseIndex:
    def __init__(self, chunks: Sequence[Chunk], model: Any, *, batch_size: int = 32) -> None:
        self.chunks = list(chunks)
        self.model = model
        self.vectors = encode_chunks(self.chunks, model, batch_size=batch_size)

    def search(self, query: str) -> list[RankedChunk]:
        if not self.chunks:
            return []
        vector = np.asarray(
            self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True), dtype=np.float32
        )[0]
        scores = self.vectors @ vector
        return _sort_scored(self.chunks, {index: float(score) for index, score in enumerate(scores)})


def reciprocal_rank_fusion(
    bm25: Sequence[RankedChunk], dense: Sequence[RankedChunk], *, constant: int = 60, depth: int = 100
) -> list[RankedChunk]:
    candidates: dict[str, Chunk] = {}
    scores: dict[str, float] = defaultdict(float)
    for ranking in (bm25, dense):
        for item in ranking[:depth]:
            candidates[item.chunk.chunk_id] = item.chunk
            scores[item.chunk.chunk_id] += 1 / (constant + item.rank)
    ordered = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], candidates[chunk_id].doc_id, chunk_id))
    return [RankedChunk(candidates[chunk_id], scores[chunk_id], rank) for rank, chunk_id in enumerate(ordered, 1)]


def rank_documents(ranking: Sequence[RankedChunk]) -> list[str]:
    """A document inherits the rank of its first chunk; later chunks are ignored."""
    return list(dict.fromkeys(item.chunk.doc_id for item in ranking))
