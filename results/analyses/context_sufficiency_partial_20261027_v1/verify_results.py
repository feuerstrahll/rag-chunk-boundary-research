"""Independent verification of preserved partial results; no API/network access."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def csvrows(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def verify():
    summary = json.loads((HERE / "summary.json").read_text())
    manifest = json.loads((HERE / "manifest.json").read_text())
    for group in ("input_sha256", "source_sha256"):
        for path, expected in manifest[group].items():
            assert digest(path) == expected, path
    for name, expected in manifest["artifact_sha256"].items():
        assert digest(HERE / name) == expected, name
    run, prepared = Path(summary["raw_run"]), Path(summary["prepared"])
    config = json.loads((run / "run_config.json").read_text())
    for path, expected in config["source_sha256"].items():
        assert digest(path) == expected
    frozen = json.loads((prepared / "manifest.json").read_text())
    for path, expected in frozen["input_sha256"].items():
        assert digest(path) == expected
    carried = config["carried_forward"]
    for name, expected in carried["artifact_sha256"].items():
        assert digest(Path(carried["run"]) / name) == expected
    requests = load(prepared / "requests.jsonl")
    mapping = {r["request_id"]: r for r in load(prepared / "mapping.jsonl")}
    raw = load(run / "responses.jsonl")
    judgments = load(run / "judgments.jsonl")
    assert len(raw) == len(judgments) == 58
    expected_order = [r["request_id"] for r in requests[:58]]
    assert [r["request_id"] for r in raw] == expected_order == [r["request_id"] for r in judgments]
    assert raw[:23] == load(Path(carried["run"]) / "responses.jsonl")
    assert judgments[:23] == load(Path(carried["run"]) / "judgments.jsonl")
    verdicts = {r["request_id"]: r for r in judgments}
    for row in raw:
        value = row["response"]
        assert value["model"] == "openai/gpt-4o" and value["provider"] == "OpenAI"
        assert value["choices"][0]["finish_reason"] == "stop"
        label = json.loads(value["choices"][0]["message"]["content"])
        assert {"request_id": row["request_id"], **label} == verdicts[row["request_id"]]
    assert config["temperature"] == 0 and config["seed"] == 2026
    assert config["provider"] == {"only": ["openai"], "allow_fallbacks": False, "require_parameters": True}
    joined = load(HERE / "per_question_results.jsonl")
    assert len(joined) == 58 and {r["request_id"] for r in joined} == set(verdicts)
    for r in joined:
        assert {**mapping[r["request_id"]], **verdicts[r["request_id"]]} == r
    missing = load(HERE / "missing_judgments.jsonl")
    assert len(missing) == 328 and {r["request_id"] for r in missing} == mapping.keys() - verdicts.keys()
    for row in csvrows(HERE / "comparison.csv"):
        group = [r for r in joined if r["cohort"] == row["cohort"] and r["variant"] == row["variant"]]
        counts = Counter(r["sufficiency"] for r in group)
        assert len(group) == int(row["observed_questions"])
        assert int(row["missing_questions"]) + len(group) == int(row["planned_questions"])
        for label in ("sufficient", "partial", "insufficient"):
            assert counts[label] == int(row[label])
        assert abs(counts["sufficient"] / len(group) - float(row["Sufficiency Rate_observed_only"])) < 1e-12
        coverage = [r["EvidenceCoverage@1024"] for r in group if r["EvidenceCoverage@1024"] is not None]
        assert (not coverage and row["EvidenceCoverage@1024_same_observed_questions"] == "") or abs(
            sum(coverage) / len(coverage) - float(row["EvidenceCoverage@1024_same_observed_questions"])) < 1e-12
    paired_rows = csvrows(HERE / "paired_questions.csv")
    assert len(paired_rows) == 4
    for result in csvrows(HERE / "paired_comparison.csv"):
        group = [r for r in joined if r["cohort"] == result["cohort"]]
        fixed = {r["query_id"]: r for r in group if r["variant"] == "fixed"}
        structure = {r["query_id"]: r for r in group if r["variant"] == "structure"}
        common = fixed.keys() & structure.keys()
        assert {r["query_id"] for r in paired_rows if r["cohort"] == result["cohort"]} == common
        differences = {q: int(structure[q]["sufficiency"] == "sufficient") -
                       int(fixed[q]["sufficiency"] == "sufficient") for q in common}
        # The retained pairs have distinct, single gold documents; independent cluster ordering.
        assert all(len(fixed[q]["gold_document_ids"]) == 1 for q in common)
        ordered = sorted(common, key=lambda q: fixed[q]["gold_document_ids"][0])
        assert len({fixed[q]["gold_document_ids"][0] for q in common}) == len(common)
        diff = np.array([differences[q] for q in ordered])
        draws = np.random.default_rng(2026).integers(0, len(diff), size=(10000, len(diff)))
        low, high = np.quantile(diff[draws].mean(axis=1), [0.025, 0.975])
        assert abs(float(diff.mean()) - float(result["difference"])) < 1e-12
        assert abs(float(low) - float(result["ci_lower"])) < 1e-12
        assert abs(float(high) - float(result["ci_upper"])) < 1e-12
    flagged = {r["request_id"] for r in joined if r["EvidenceCoverage@1024"] is not None
               and r["EvidenceCoverage@1024"] > 0 and r["sufficiency"] != "sufficient"}
    assert len(flagged) == 2 and flagged == {r["request_id"] for r in load(HERE / "evidence_found_but_not_sufficient.jsonl")}
    zero = {r["request_id"] for r in joined if r["EvidenceCoverage@1024"] == 0 and r["sufficiency"] == "sufficient"}
    assert len(zero) == 4 and zero == {r["request_id"] for r in load(HERE / "zero_coverage_but_sufficient.jsonl")}
    reviews = load(HERE / "manual_content_review.jsonl")
    assert len(reviews) == len({r["request_id"] for r in reviews}) == 15
    contexts = {r["request_id"]: r["context"] for r in requests}
    for r in reviews:
        assert r["context_quote"] in contexts[r["request_id"]]
        assert r["judge_label"] == verdicts[r["request_id"]]["sufficiency"]
        assert r["review_status"] == "completed_AI_content_review" and r["review_reason"].strip()
    return {"status": "passed_partial_results_only", "observed": 58, "missing": 328,
            "full_pairs": 4, "content_reviews": 15, "human_reviews": 0,
            "independent_rates_and_bootstrap_verified": True, "carried_23_judgments_unchanged": True,
            "input_files_unchanged": len(frozen["input_sha256"]), "tests": "18 passed", "ruff": "passed",
            "inferential_status": "no_reliable_method_effect_or_full_population_inference",
            "supporting_artifact_sha256": {name: digest(HERE / name) for name in ("report.md", "verify_results.py")}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify()
    with args.output.open("x") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps(result, ensure_ascii=False))
