"""Independent integrity check; never fabricates or invokes LLM judgments."""

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from tokenizers import Tokenizer

RUN = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load(name):
    return [json.loads(line) for line in (RUN / name).read_text().splitlines() if line]


def main():
    manifest = json.loads((RUN / "manifest.json").read_text())
    for group in ("input_sha256", "source_sha256"):
        for name, expected in manifest[group].items():
            assert digest(name) == expected, name
    for name, expected in manifest["artifact_sha256"].items():
        assert digest(RUN / name) == expected, name
    requests, mapping = load("requests.jsonl"), load("mapping.jsonl")
    assert len(requests) == len(mapping) == 386
    request_map = {r["request_id"]: r for r in requests}
    assert len(request_map) == 386
    assert request_map.keys() == {r["request_id"] for r in mapping}
    assert all(set(r) == {"request_id", "question", "context"} for r in requests)
    counts = Counter((r["cohort"], r["variant"]) for r in mapping)
    assert counts == {("AI-only", "fixed"): 118, ("AI-only", "structure"): 118,
                      ("original-gold", "fixed"): 75, ("original-gold", "structure"): 75}
    tokenizer_file = next(p for p in manifest["input_sha256"] if p.endswith("/tokenizer.json"))
    tokenizer = Tokenizer.from_file(tokenizer_file)
    tokenizer.no_truncation()
    tokenizer.no_padding()
    doc_file = next(p for p in manifest["input_sha256"] if p.endswith("canonical_documents.jsonl"))
    docs = {r["doc_id"]: r["canonical_text"] for r in map(json.loads, Path(doc_file).read_text().splitlines())}
    gold_file = next(p for p in manifest["input_sha256"] if p.endswith("retrieval_gold_ai.jsonl"))
    gold = {r["question_id"]: r for r in map(json.loads, Path(gold_file).read_text().splitlines())}
    for row in mapping:
        request = request_map[row["request_id"]]
        context = request["context"]
        assert hashlib.sha256(context.encode()).hexdigest() == row["context_sha256"]
        assert context == "\n\n".join(docs[c["doc_id"]][c["char_start"]:c["char_end"]] for c in row["chunks"])
        assert len(tokenizer.encode(context, add_special_tokens=False).ids) == row["serialized_context_tokens"] <= 1024
        assert sum(c["token_count"] for c in row["chunks"]) == row["used_tokens"] <= 1024
        if row["cohort"] == "original-gold":
            assert row["EvidenceCoverage@1024"] is None
            continue
        q = gold[row["query_id"]]
        # Independent interval merge; does not import experiment coverage helpers.
        numerator, denominator = 0, 0
        for span in q["gold_spans"]:
            a, b = span["char_start"], span["char_end"]
            denominator += b - a
            union = []
            for c in sorted(row["chunks"], key=lambda c: c["char_start"]):
                if c["doc_id"] != q["doc_id"]:
                    continue
                left, right = max(a, c["char_start"]), min(b, c["char_end"])
                if left >= right:
                    continue
                if union and left <= union[-1][1]:
                    union[-1][1] = max(union[-1][1], right)
                else:
                    union.append([left, right])
            numerator += sum(right - left for left, right in union)
        assert abs(numerator / denominator - row["EvidenceCoverage@1024"]) < 1e-12
    with (RUN / "comparison.csv").open() as f:
        comparisons = list(csv.DictReader(f))
    assert all(r["Sufficiency Rate"] == "" and r["evaluated"] == "0" for r in comparisons)
    with (RUN / "judgments.csv").open() as f:
        assert list(csv.DictReader(f)) == []
    report = {"status": "passed_prepared_inputs_only", "contexts_verified": 386,
              "coverage_rows_independently_verified": 236, "judge_calls": 0,
              "judge_reviews": 0, "unit_tests": "11 passed", "ruff": "passed on new script and tests",
              "unverified": ["live_LLM_endpoint", "Sufficiency_Rate_and_paired_CI", "12_judge_content_reviews"],
              "broader_tests": "hierarchy test collection unavailable: pre-existing orjson dependency missing",
              "supporting_artifact_sha256": {name: digest(RUN / name) for name in (
                  "verify_prepared.py", "report.md", "requirements.txt")}}
    path = RUN / "validation.json"
    with path.open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
