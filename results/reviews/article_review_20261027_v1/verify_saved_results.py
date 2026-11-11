"""Independent standard-library audit; does not import project metric helpers.

Run: python article/results/reviews/article_review_20261027_v1/verify_saved_results.py --output /tmp/article-review-checks.json
This replays saved B1024 contexts, not retrieval, tokenization or full rankings.
"""
from pathlib import Path
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import platform
import random
import statistics

ARTICLE = Path(__file__).resolve().parents[3]
RUN = ARTICLE / "results/runs/hierarchy_full_20261026_v1"
POST = ARTICLE / "results/analyses/hierarchy_posthoc_20261026_v2"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def coverage(question, chunks):
    numerator = 0
    denominator = 0
    for span in question["gold_spans"]:
        start, end = span["char_start"], span["char_end"]
        denominator += end - start
        intersections = sorted(
            (max(start, c["char_start"]), min(end, c["char_end"]))
            for c in chunks if c["doc_id"] == question["doc_id"]
        )
        right = start
        for a, b in intersections:
            if b > max(a, right):
                numerator += b - max(a, right)
                right = b
    return numerator / denominator


def audit():
    checked = {}
    for folder in [RUN, POST]:
        manifest = read(folder / "manifest.json")
        for relative, expected in manifest["artifact_sha256"].items():
            path = folder / relative
            assert sha(path) == expected, path
        checked[str(folder.relative_to(ARTICLE))] = len(manifest["artifact_sha256"])
    config = read(RUN / "frozen_config.json")
    preflight = read(RUN / "preflight.json")
    for name, expected in preflight["source_hashes"].items():
        assert sha(Path(name)) == expected, name
    model = Path(config["model"]["local_snapshot"])
    for name, expected in preflight["model_hashes"].items():
        assert sha(model / name) == expected, name
    corpus = (ARTICLE / config["dataset"]["corpus"]).resolve()
    gold_path = (ARTICLE / config["gold"]["path"]).resolve()
    assert sha(corpus) == config["dataset"]["corpus_sha256"]
    assert sha(gold_path) == config["gold"]["sha256"]
    gold = {q["question_id"]: q for q in rows(gold_path)}
    assert len(gold) == 118
    assert sum(len(q["gold_spans"]) for q in gold.values()) == 147
    assert len({q["doc_id"] for q in gold.values()}) == 118
    for q in gold.values():
        spans = sorted(q["gold_spans"], key=lambda s: s["char_start"])
        assert all(a["char_end"] <= b["char_start"] for a, b in zip(spans, spans[1:]))
    aggregates = {(r["variant"], r["method"]): r for r in csv.DictReader((RUN / "aggregate_metrics.csv").open())}
    values, recalculated, maximum, depth = {}, [], 0, {}
    for variant in config["variants"]:
        contexts = rows(RUN / variant / "contexts.jsonl")
        assert len(contexts) == 354
        assert len({(c["query_id"], c["method"]) for c in contexts}) == 354
        for context in contexts:
            used = sum(c["token_count"] for c in context["chunks"])
            assert used == context["used_tokens"] and used <= 1024
            maximum = max(maximum, used)
            key = variant, context["method"], context["query_id"]
            values[key] = coverage(gold[context["query_id"]], context["chunks"])
        for method in config["retrieval"]["methods"]:
            mean = statistics.mean(values[variant, method, q] for q in gold)
            expected = float(aggregates[variant, method]["EvidenceCoverage@B=1024"])
            assert abs(mean - expected) < 1e-12
            recalculated.append({"variant": variant, "retriever": method, "Coverage@B1024": mean})
            depth[f"{variant}/{method}"] = max(c["rank"] for x in contexts if x["method"] == method for c in x["chunks"])
    lengths = {r["doc_id"]: r for r in csv.DictReader((POST / "document_lengths.csv").open())}
    bands = []
    for band in ["<=510", "511-1024", "1025-2048", ">2048"]:
        ids = [q for q in gold if lengths[gold[q]["doc_id"]]["length_band"] == band]
        bands.append({"band": band, "documents": sum(x["length_band"] == band for x in lengths.values()),
                      "questions": len(ids), "hybrid_coverage": {
                          v: statistics.mean(values[v, "hybrid", q] for q in ids) for v in config["variants"]}})
    paired = []
    for reference in ["structure", "fixed", "unrestricted"]:
        differences = [values["hierarchy", "hybrid", q] - values[reference, "hybrid", q] for q in sorted(gold)]
        rng = random.Random(2026)
        draws = sorted(statistics.mean(rng.choices(differences, k=118)) for _ in range(10000))
        def quantile(p):
            position = (len(draws) - 1) * p
            i = int(position)
            return draws[i] + (position - i) * (draws[min(i + 1, len(draws) - 1)] - draws[i])
        paired.append({"reference": reference, "difference": statistics.mean(differences),
                       "wins": sum(x > 1e-12 for x in differences), "losses": sum(x < -1e-12 for x in differences),
                       "ties": sum(abs(x) <= 1e-12 for x in differences),
                       "independent_bootstrap_95pct": [quantile(.025), quantile(.975)]})
    headings = rows(POST / "heading_cases.jsonl")
    heading_counts = {
        "records": len(headings), "documents": len({h["doc_id"] for h in headings}),
        "heading_only": sum(h["heading_chunk_contains_only_headings"] for h in headings),
        "other_body": sum(h["heading_chunk_has_other_body_after_comment_exclusion"] for h in headings),
        "headings_and_only_comments": sum(not h["heading_chunk_contains_only_headings"] and not h["heading_chunk_has_other_body_after_comment_exclusion"] for h in headings),
        "heading_body_fits_target": sum(bool(h["full_heading_body_span_fits_target"]) for h in headings),
        "ablation_preserves_pair": sum(bool(h["ablation_contains_heading_and_full_first_body"]) for h in headings),
    }
    article = next(ARTICLE.glob("*.docx"))
    return {"status": "passed_with_review_limits", "result_type": "new_diagnostic_audit",
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
            "reviewed_docx": article.name, "reviewed_docx_sha256": sha(article),
            "verification_script_sha256": sha(Path(__file__)), "artifact_hash_counts": checked,
            "preflight_source_hashes_checked": len(preflight["source_hashes"]),
            "model_hashes_checked": len(preflight["model_hashes"]),
            "corpus_sha256": sha(corpus), "gold_sha256": sha(gold_path),
            "saved_contexts_replayed": len(values), "maximum_context_tokens": maximum,
            "coverage_aggregates": recalculated, "length_bands": bands,
            "paired_hybrid_comparisons": paired, "heading_counts": heading_counts,
            "maximum_selected_rank": depth,
            "limits": ["No end-to-end encoder/retrieval rerun; no fresh pytest run.",
                       "No independent replay of top-5 or document metrics from Parquet in this environment.",
                       "Independent bootstrap uses Python Random, not the original NumPy RNG; compare interpretation, not bitwise endpoints.",
                       "Recorded token counts checked arithmetically; tokenizer not rerun.",
                       "No independent human judgment of semantic label validity or DOCX visual rendering."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    output = parser.parse_args().output
    if output.exists():
        raise FileExistsError(output)
    result = audit()
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "contexts": result["saved_contexts_replayed"],
                      "hashes": result["artifact_hash_counts"], "output": str(output)}, ensure_ascii=False))
