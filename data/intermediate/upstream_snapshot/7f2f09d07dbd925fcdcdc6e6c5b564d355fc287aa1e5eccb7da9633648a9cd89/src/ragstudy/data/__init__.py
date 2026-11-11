"""Dataset-owned interfaces, not dataset-specific transformations."""

from .interfaces import (
    DocumentSource,
    GoldLabels,
    LayoutGold,
    ParsingDatasetLoader,
    Query,
    RetrievalDatasetLoader,
)
from .watsonx_docsqa import WATSONX_REPRESENTATIONS, WatsonxDocsQACorpusLoader
from .watsonx_qa import GoldEvidence, QARecord, load_qa_parquets

__all__ = [
    "DocumentSource",
    "GoldEvidence",
    "GoldLabels",
    "LayoutGold",
    "ParsingDatasetLoader",
    "Query",
    "QARecord",
    "RetrievalDatasetLoader",
    "WATSONX_REPRESENTATIONS",
    "WatsonxDocsQACorpusLoader",
    "load_qa_parquets",
]
