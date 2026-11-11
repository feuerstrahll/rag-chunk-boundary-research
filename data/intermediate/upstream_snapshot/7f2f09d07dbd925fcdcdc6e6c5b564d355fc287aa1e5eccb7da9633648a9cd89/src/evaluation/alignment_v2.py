"""Explicit v2 alignment entry point for the isolated HTML experiment."""

from __future__ import annotations

from models import Document
from ragstudy.data.watsonx_qa import GoldEvidence

from .retrieval import AlignedEvidence, align_evidence

EVIDENCE_ALIGNER_VERSION = "evidence_aligner_v2"


def align_evidence_v2(evidence: GoldEvidence, document: Document) -> AlignedEvidence:
    """Run the frozen aligner against v2 parser output.

    No new fuzzy-alignment rule was approved for stage 5; the separate entry
    point makes the parser effect measurable without conflating both changes.
    """
    return align_evidence(evidence, document)
