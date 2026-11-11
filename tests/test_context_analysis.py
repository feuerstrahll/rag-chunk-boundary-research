"""Check the independent report validator on a nontrivial budget skip."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyze_context_packing import independent_metric_check


def test_independent_metric_reconstruction_detects_wrong_coverage(tmp_path):
    chunks = [
        {"chunk_id": "a", "doc_id": "gold", "char_start": 0, "char_end": 40, "token_count": 480},
        {"chunk_id": "b", "doc_id": "other", "char_start": 0, "char_end": 100, "token_count": 480},
        {"chunk_id": "c", "doc_id": "gold", "char_start": 40, "char_end": 100, "token_count": 300},
        {"chunk_id": "d", "doc_id": "third", "char_start": 0, "char_end": 20, "token_count": 50},
    ]
    ranking = [{"query_id": "q", "method": "bm25", "rank": i + 1, "chunk_index": i, "score": 1 / (i + 1)}
               for i in range(4)]
    pq.write_table(pa.Table.from_pylist(ranking), tmp_path / "rankings.parquet")
    (tmp_path / "contexts.jsonl").write_text(json.dumps({"query_id": "q", "method": "bm25",
                                         "used_tokens": 1010, "chunks": [chunks[i] for i in [0, 1, 3]]}) + "\n")
    gold = [{"question_id": "q", "doc_id": "gold",
             "gold_spans": [{"char_start": 0, "char_end": 40}, {"char_start": 60, "char_end": 100}]}]
    row = {"query_id": "q", "method": "bm25", "EvidenceCoverage@B=1024": .5, "EvidenceCoverage@5": 1.,
           "EvidenceRecall@5": 1., "EvidenceMRR@5": 0., "DocumentRecall@5": 1., "DocumentMRR@10": 1.,
           "actual_context_tokens_B1024": 1010}
    _, top, receipt = independent_metric_check(tmp_path, chunks, gold, [row])
    assert receipt["query_method_rows"] == 1 and len(top["q", "bm25"]) == 4
    with pytest.raises(ValueError, match="Independent metric mismatch"):
        independent_metric_check(tmp_path, chunks, gold, [{**row, "EvidenceCoverage@B=1024": 1.}])
