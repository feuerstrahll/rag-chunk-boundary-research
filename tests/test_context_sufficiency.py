"""Synthetic unit fixtures only; never used as experimental LLM judgments."""

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/evaluate_context_sufficiency.py"
spec = importlib.util.spec_from_file_location("context_sufficiency", SCRIPT)
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def test_judge_payload_has_only_question_and_context():
    payload = evaluation.messages({"request_id": "opaque", "question": "Q", "context": "C",
                                   "reference_answer": "SECRET", "variant": "fixed"})
    assert json.loads(payload[1]["content"]) == {"question": "Q", "context": "C"}
    assert "SECRET" not in json.dumps(payload)
    assert "fixed" not in json.dumps(payload)


@pytest.mark.parametrize("label,missing", [("sufficient", ""), ("partial", "second part"),
                                           ("insufficient", "relevant facts")])
def test_valid_labels(label, missing):
    assert evaluation.validate_judgment({"sufficiency": label, "reason": "reason",
                                         "missing_information": missing})["sufficiency"] == label


@pytest.mark.parametrize("value", [
    {"sufficiency": "yes", "reason": "reason", "missing_information": ""},
    {"sufficiency": "partial", "reason": "reason", "missing_information": ""},
    {"sufficiency": "sufficient", "reason": "reason", "missing_information": "missing"},
    {"sufficiency": "sufficient", "reason": "", "missing_information": ""},
])
def test_malformed_judgments_fail_closed(value):
    with pytest.raises(ValueError):
        evaluation.validate_judgment(value)


def test_unavailable_is_not_zero_or_insufficient():
    mapping = [{"cohort": "AI-only", "variant": v, "request_id": v,
                "EvidenceCoverage@1024": 1.0} for v in ("fixed", "structure")]
    for row in evaluation.comparison(mapping, {}):
        assert row["evaluated"] == 0
        assert row["sufficient"] is None
        assert row["Sufficiency Rate"] is None


def test_analysis_pairs_and_disagreements(tmp_path):
    prepared, output = tmp_path / "prepared", tmp_path / "output"
    prepared.mkdir()
    output.mkdir()
    mapping, requests, judgments = [], [], []
    for q in range(12):
        for variant in ("fixed", "structure"):
            rid = f"unit_fixture_{q}_{variant}"
            mapping.append({"cohort": "AI-only", "query_id": str(q), "variant": variant,
                            "request_id": rid, "gold_document_ids": [str(q)],
                            "EvidenceCoverage@1024": 1.0})
            requests.append({"request_id": rid, "question": "fixture", "context": "fixture"})
            judgments.append({"request_id": rid, "sufficiency": "sufficient" if variant == "structure" else "partial",
                              "reason": "unit fixture", "missing_information": "" if variant == "structure" else "fixture"})
    evaluation.jsonl(prepared / "mapping.jsonl", mapping)
    evaluation.jsonl(prepared / "requests.jsonl", requests)
    evaluation.jsonl(output / "judgments.jsonl", judgments)
    evaluation.analyze(prepared, output)
    import csv

    with (output / "paired_comparison.csv").open() as f:
        pair = next(csv.DictReader(f))
    assert float(pair["difference"]) == float(pair["ci_lower"]) == float(pair["ci_upper"]) == 1
    assert len(evaluation.rows(output / "evidence_found_but_not_sufficient.jsonl")) == 12
    reviews = evaluation.rows(output / "manual_review_packet.jsonl")
    assert len(reviews) == 12
    assert all(r["review_status"] == "pending" for r in reviews)


def test_incomplete_pairs_cannot_produce_rates(tmp_path):
    prepared, output = tmp_path / "prepared", tmp_path / "output"
    prepared.mkdir()
    output.mkdir()
    evaluation.jsonl(prepared / "mapping.jsonl", [])
    evaluation.jsonl(prepared / "requests.jsonl", [{"request_id": "missing"}])
    evaluation.jsonl(output / "judgments.jsonl", [])
    with pytest.raises(ValueError, match="every prepared request"):
        evaluation.analyze(prepared, output)
