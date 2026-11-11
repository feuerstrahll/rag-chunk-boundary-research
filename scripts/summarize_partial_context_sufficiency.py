"""Preserve an explicitly stopped, incomplete sufficiency experiment without imputation."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean

import evaluate_context_sufficiency as evaluation


def summarize(args):
    from article_research.packing_evaluation import (
        document_clusters,
        paired_cluster_bootstrap,
    )

    run, prepared = args.run.resolve(), args.prepared.resolve()
    config = evaluation.read(run / "run_config.json")
    for name, expected in config["input_sha256"].items():
        if evaluation.sha(prepared / name) != expected:
            raise ValueError("Frozen context inputs changed")
    values = evaluation.rows(run / "judgments.jsonl")
    raw = evaluation.rows(run / "responses.jsonl")
    mapping = evaluation.rows(prepared / "mapping.jsonl")
    judgments = {r["request_id"]: r for r in values}
    raw_map = {r["request_id"]: r["response"] for r in raw}
    all_ids = {r["request_id"] for r in mapping}
    if not (0 < len(judgments) == len(values) == len(raw_map) == len(raw) < len(mapping)):
        raise ValueError("Expected unique, genuinely partial raw/scored responses")
    if not judgments.keys() == raw_map.keys() <= all_ids:
        raise ValueError("Partial query IDs do not match")
    for rid, row in judgments.items():
        body = raw_map[rid]
        if body["model"] != "openai/gpt-4o" or body["provider"] != "OpenAI":
            raise ValueError("Unexpected model/provider")
        choice = body["choices"][0]
        if choice["finish_reason"] != "stop":
            raise ValueError("Truncated output")
        verdict = evaluation.validate_judgment(json.loads(choice["message"]["content"]))
        if {"request_id": rid, **verdict} != row:
            raise ValueError("Raw/scored labels differ")
    joined = [{**r, **judgments[r["request_id"]]} for r in mapping if r["request_id"] in judgments]
    comparisons, pairs, pair_rows = [], [], []
    for cohort in sorted({r["cohort"] for r in mapping}):
        scores, docs = {"fixed": {}, "structure": {}}, {}
        for variant in ("fixed", "structure"):
            group = [r for r in joined if r["cohort"] == cohort and r["variant"] == variant]
            labels = Counter(r["sufficiency"] for r in group)
            planned = sum(r["cohort"] == cohort and r["variant"] == variant for r in mapping)
            coverages = [r["EvidenceCoverage@1024"] for r in group if r["EvidenceCoverage@1024"] is not None]
            comparisons.append({"cohort": cohort, "variant": variant, "planned_questions": planned,
                "observed_questions": len(group), "missing_questions": planned - len(group),
                "sufficient": labels["sufficient"], "partial": labels["partial"], "insufficient": labels["insufficient"],
                "Sufficiency Rate_observed_only": labels["sufficient"] / len(group) if group else None,
                "EvidenceCoverage@1024_same_observed_questions": mean(coverages) if coverages else None,
                "scope": "incomplete_unpaired_observed_subset_not_method_effect"})
            for r in group:
                scores[variant][r["query_id"]] = float(r["sufficiency"] == "sufficient")
                docs[r["query_id"]] = r["gold_document_ids"]
        common = scores["fixed"].keys() & scores["structure"].keys()
        if not common:
            continue
        reference = {q: scores["fixed"][q] for q in sorted(common)}
        candidate = {q: scores["structure"][q] for q in sorted(common)}
        clusters = document_clusters({q: docs[q] for q in sorted(common)})
        stats = paired_cluster_bootstrap(reference, candidate, clusters)
        pairs.append({"cohort": cohort, "contrast": "structure_minus_fixed",
                      "fixed_rate_paired_subset": mean(reference.values()),
                      "structure_rate_paired_subset": mean(candidate.values()), **stats,
                      "inferential_status": "mechanical_bootstrap_only_tiny_incomplete_sample_no_reliable_inference"})
        pair_rows.extend({"cohort": cohort, "query_id": q, "fixed_sufficient": reference[q],
                          "structure_sufficient": candidate[q], "difference": candidate[q] - reference[q]} for q in sorted(common))
    output = evaluation.fresh(args.output)
    evaluation.table(output / "comparison.csv", comparisons)
    evaluation.table(output / "paired_comparison.csv", pairs)
    evaluation.table(output / "paired_questions.csv", pair_rows)
    evaluation.table(output / "judgments.csv", values)
    evaluation.jsonl(output / "per_question_results.jsonl", joined)
    missing = [{"request_id": r["request_id"], "cohort": r["cohort"], "query_id": r["query_id"],
                "variant": r["variant"], "status": "not_evaluated_no_imputation"} for r in mapping if r["request_id"] not in judgments]
    evaluation.jsonl(output / "missing_judgments.jsonl", missing)
    flagged = [r for r in joined if r["EvidenceCoverage@1024"] is not None
               and r["EvidenceCoverage@1024"] > 0 and r["sufficiency"] != "sufficient"]
    evaluation.jsonl(output / "evidence_found_but_not_sufficient.jsonl", flagged)
    zero_sufficient = [r for r in joined if r["EvidenceCoverage@1024"] == 0 and r["sufficiency"] == "sufficient"]
    evaluation.jsonl(output / "zero_coverage_but_sufficient.jsonl", zero_sufficient)
    cross = Counter((r["variant"], "zero" if r["EvidenceCoverage@1024"] == 0 else
                     "full" if r["EvidenceCoverage@1024"] == 1 else "partial", r["sufficiency"])
                    for r in joined if r["EvidenceCoverage@1024"] is not None)
    evaluation.table(output / "coverage_sufficiency_counts.csv", [
        {"variant": v, "coverage_band": b, "sufficiency": label, "count": cross[v, b, label]}
        for v in ("fixed", "structure") for b in ("zero", "partial", "full")
        for label in ("sufficient", "partial", "insufficient")])
    reviews = evaluation.rows(args.review)
    if not 10 <= len(reviews) <= 15 or not {r["request_id"] for r in reviews} <= judgments.keys():
        raise ValueError("Review sample does not match received judgments")
    evaluation.jsonl(output / "manual_content_review.jsonl", reviews)
    evaluation.table(output / "manual_content_review.csv", reviews)
    summary = {"status": "completed_partial_at_user_request", "planned_contexts": len(mapping),
        "observed_contexts": len(joined), "missing_contexts": len(missing), "paired_questions": len(pair_rows),
        "content_reviews": len(reviews), "review_type": "AI_content_review_not_human_annotation",
        "human_reviews": 0, "judge": "openai/gpt-4o", "provider": "OpenAI", "temperature": 0,
        "full_population_rates_or_method_superiority_estimated": False,
        "stop_reason": "user_requested_preservation_after_OpenRouter_credit_error",
        "comparisons": comparisons, "paired_diagnostics": pairs,
        "coverage_positive_non_sufficient": len(flagged), "zero_coverage_sufficient": len(zero_sufficient),
        "prepared": str(prepared), "raw_run": str(run)}
    evaluation.write(output / "summary.json", summary)
    evaluation.write(output / "manifest.json", {
        "status": summary["status"], "command": __import__("sys").argv,
        "input_sha256": {str(p): evaluation.sha(p) for p in (
            prepared / "manifest.json", prepared / "mapping.jsonl", prepared / "requests.jsonl",
            run / "run_config.json", run / "responses.jsonl", run / "judgments.jsonl", args.review.resolve())},
        "source_sha256": {str(p): evaluation.sha(p) for p in (Path(__file__).resolve(),
            Path(evaluation.__file__).resolve(), evaluation.ARTICLE / "src/article_research/packing_evaluation.py")},
        "artifact_sha256": {p.name: evaluation.sha(p) for p in output.iterdir() if p.is_file()}})
    print(json.dumps(summary, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    summarize(parser.parse_args())


if __name__ == "__main__":
    main()
