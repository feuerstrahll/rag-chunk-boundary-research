"""Corpus-only index-budget matching for controlled chunking comparisons."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class BudgetConstraints:
    min_tokens: int
    max_tokens: int
    max_overlap_tokens: int
    max_overlap_fraction: float
    relative_tolerance: float
    shortlist_size: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "BudgetConstraints":
        constraints = cls(
            min_tokens=int(value["min_tokens"]),
            max_tokens=int(value["max_tokens"]),
            max_overlap_tokens=int(value["max_overlap_tokens"]),
            max_overlap_fraction=float(value["max_overlap_fraction"]),
            relative_tolerance=float(value["relative_tolerance"]),
            shortlist_size=int(value["shortlist_size"]),
        )
        constraints.validate()
        return constraints

    def validate(self) -> None:
        if self.min_tokens < 1 or self.max_tokens < self.min_tokens:
            raise ValueError("Budget matching requires an ordered positive token range")
        if self.max_overlap_tokens < 0:
            raise ValueError("max_overlap_tokens must not be negative")
        if not 0 <= self.max_overlap_fraction <= 0.5:
            raise ValueError("max_overlap_fraction must be between zero and 0.5")
        if not 0 < self.relative_tolerance < 1:
            raise ValueError("relative_tolerance must be between zero and one")
        if self.shortlist_size < 1:
            raise ValueError("shortlist_size must be positive")


@dataclass(frozen=True)
class BudgetCandidate:
    max_tokens: int
    overlap_tokens: int
    chunk_count: int
    total_tokens: int
    chunk_count_gap: float
    total_tokens_gap: float
    objective: float
    pathological_match: bool
    pathological_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["pathological_reasons"] = list(self.pathological_reasons)
        return result


def pathological_reasons(
    max_tokens: int, overlap_tokens: int, constraints: BudgetConstraints
) -> tuple[str, ...]:
    reasons: list[str] = []
    if not constraints.min_tokens <= max_tokens <= constraints.max_tokens:
        reasons.append("max_tokens_outside_admissible_range")
    if not 0 <= overlap_tokens < max_tokens:
        reasons.append("invalid_overlap")
    if overlap_tokens > constraints.max_overlap_tokens:
        reasons.append("overlap_exceeds_absolute_limit")
    if overlap_tokens / max_tokens > constraints.max_overlap_fraction:
        reasons.append("overlap_exceeds_fractional_limit")
    return tuple(reasons)


def make_candidate(
    *,
    max_tokens: int,
    overlap_tokens: int,
    chunk_count: int,
    total_tokens: int,
    target_chunk_count: int,
    target_total_tokens: int,
    constraints: BudgetConstraints,
) -> BudgetCandidate:
    if target_chunk_count <= 0 or target_total_tokens <= 0:
        raise ValueError("Target budget statistics must be positive")
    count_gap = abs(chunk_count - target_chunk_count) / target_chunk_count
    token_gap = abs(total_tokens - target_total_tokens) / target_total_tokens
    reasons = pathological_reasons(max_tokens, overlap_tokens, constraints)
    return BudgetCandidate(
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        chunk_count=chunk_count,
        total_tokens=total_tokens,
        chunk_count_gap=count_gap,
        total_tokens_gap=token_gap,
        objective=max(count_gap, token_gap),
        pathological_match=bool(reasons),
        pathological_reasons=reasons,
    )


def _estimated_budget(length_counts: Counter[int], max_tokens: int, overlap_tokens: int) -> tuple[int, int]:
    stride = max_tokens - overlap_tokens
    chunks = total_tokens = 0
    for length, documents in length_counts.items():
        if length <= 0:
            continue
        count = 1 if length <= max_tokens else 1 + math.ceil((length - max_tokens) / stride)
        final_length = length - (count - 1) * stride
        emitted = (count - 1) * max_tokens + final_length
        chunks += documents * count
        total_tokens += documents * emitted
    return chunks, total_tokens


def shortlist_budget_candidates(
    document_token_lengths: Sequence[int],
    *,
    target_chunk_count: int,
    target_total_tokens: int,
    constraints: BudgetConstraints,
) -> list[BudgetCandidate]:
    """Rank admissible configurations without using questions, labels, or retrieval scores."""
    if any(length < 0 for length in document_token_lengths):
        raise ValueError("Document token lengths must not be negative")
    length_counts = Counter(document_token_lengths)
    candidates: list[BudgetCandidate] = []
    for max_tokens in range(constraints.min_tokens, constraints.max_tokens + 1):
        overlap_limit = min(
            constraints.max_overlap_tokens,
            int(max_tokens * constraints.max_overlap_fraction),
            max_tokens - 1,
        )
        for overlap_tokens in range(overlap_limit + 1):
            chunk_count, total_tokens = _estimated_budget(length_counts, max_tokens, overlap_tokens)
            candidates.append(
                make_candidate(
                    max_tokens=max_tokens,
                    overlap_tokens=overlap_tokens,
                    chunk_count=chunk_count,
                    total_tokens=total_tokens,
                    target_chunk_count=target_chunk_count,
                    target_total_tokens=target_total_tokens,
                    constraints=constraints,
                )
            )
    candidates.sort(key=_candidate_order)
    return candidates[: constraints.shortlist_size]


def choose_budget_match(candidates: Sequence[BudgetCandidate], constraints: BudgetConstraints) -> BudgetCandidate:
    if not candidates:
        raise ValueError("Budget matching requires at least one evaluated candidate")
    admissible = [candidate for candidate in candidates if not candidate.pathological_match]
    if not admissible:
        raise ValueError("Budget matching produced no admissible candidate")
    return min(admissible, key=_candidate_order)


def match_status(candidate: BudgetCandidate, constraints: BudgetConstraints) -> str:
    if candidate.pathological_match:
        return "pathological"
    if (
        candidate.chunk_count_gap <= constraints.relative_tolerance
        and candidate.total_tokens_gap <= constraints.relative_tolerance
    ):
        return "matched"
    return "approximate"


def _candidate_order(candidate: BudgetCandidate) -> tuple[float, float, int, int]:
    return (
        candidate.objective,
        candidate.chunk_count_gap + candidate.total_tokens_gap,
        candidate.overlap_tokens,
        -candidate.max_tokens,
    )
