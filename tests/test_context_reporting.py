"""Check that the requested research deliverable renders all comparison parts."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from context_packing_report import render_report


def test_report_keeps_metrics_components_provenance_and_zero_denominators(tmp_path):
    variants = ["v1", "headings", "headings_intro"]
    metric_names = ["EvidenceCoverage@B=1024", "EvidenceCoverage@5", "EvidenceRecall@5",
                    "EvidenceMRR@5", "DocumentRecall@5", "DocumentMRR@10"]
    timing_names = ["segmentation_seconds", "bm25_build_seconds", "dense_build_including_encoder_validation_seconds",
                    "query_ranking_scoring_and_artifact_io_seconds"]
    costs = {v: {"chunk_count": 2, "indexed_tokens": 200, "chunk_tokens": dict.fromkeys(
                  ["min", "p10", "median", "p90", "max"], 100), "actual_overlap_tokens_sum": 0,
                  "fallback_chunks": 0, "stages_seconds": dict.fromkeys(timing_names, .1),
                  "peak_process_rss_bytes": 2**20, "dense_matrix_bytes": 1024} for v in variants}
    behavior = {v: {"all_heading_body_links": 132, "known_cases_preserved": 132, "all_preserved": 132,
                    "broken_fits_target": 0, "broken_fits_encoder": 0, "broken_exceeds_encoder": 0,
                    "new_broken_relative_to_v1": 0, "new_feasible_broken_relative_to_v1": 0} for v in variants}
    summary = {"documents": 1144, "questions": 118, "cluster_count": 80,
               "software_checks": {"full_suite": {"passed": 50, "failed": 5}},
               "aggregates": [{"variant": v, "method": m, **dict.fromkeys(metric_names, .5),
                               "mean_actual_context_tokens": 1000} for v in variants for m in ["bm25", "dense", "hybrid"]],
               "costs": costs, "heading_behavior": behavior,
               "paired_intervals": [{"contrast": c, "method": "hybrid", "metric": metric_names[0],
                                    "difference": 0, "ci_lower": 0, "ci_upper": 0}
                                    for c in ["headings_minus_v1", "headings_intro_minus_headings"]],
               "detector": {"headings_intro": {"intro_links_proposed": 0, "documents_with_intro_links": 0,
                            "intro_links_preserved": 0, "intro_links_exceed_encoder": 0,
                            "packing_decisions": {}, "abstention_reasons": {}}},
               "semantic_label_status": {"labeled_documents": 1, "evaluation_questions_in_labeled_documents": 0},
               "index_preservation": [], "span_equality": [{"reference": "headings", "candidate": "headings_intro",
                                                            "identical_spans": 1144, "documents": 1144}],
               "retrieved_completeness": [{"variant": "v1", "method": "hybrid", "context": "B1024",
                                         "cohort": "existing_content_checked_AI", "scope": "all_indexed_documents",
                                         "eligible_query_group_pairs": 826, "b_fully_returned": 0,
                                         "complete": 0, "b_returned_missing_a": 0, "complete_given_b_returned": None}]}
    examples = [{"record_id": r, "doc_id": "d", "b_quote": "Port number", "a_quote": "Watson Query connection"}
                for r in ["pilot_025", "pilot_004", "pilot_008", "pilot_010"]]
    (tmp_path / "development_examples.json").write_text(json.dumps({"examples": examples}))
    (tmp_path / "headings_intro").mkdir()
    (tmp_path / "headings_intro/preferred_links.jsonl").write_text("")
    render_report(tmp_path, summary, [], [], {}, {})
    text = (tmp_path / "report.md").read_text()
    assert text.count("| 0.500000 |") >= 9
    assert "— (B не найден)" in text
    assert "Человеческий знаменатель равен **0**" in text
    assert "## Срабатывания отключаемого правила" in text and "## Проверяемые примеры" in text
    assert "не возникло дополнительного изменения границ" in text
    assert "**50 passed, 5 failed**" in text and "отсутствующего executable git" in text
    assert "интервал" in text and "не независимое подтверждение" not in text  # zero interval takes the cautious branch
