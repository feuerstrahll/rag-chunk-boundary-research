"""Rebuild twelve explicitly AI-authored calibration suggestions, never human labels."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "results" / "pilots" / "dependency_pilot_20261022_v5"
OUT = ROOT / "results" / "pilots" / "dependency_calibration_ai_draft_20261022_v1.jsonl"


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


# A is identified by its block number and an exact substring of canonical block text.
# Empty A means no unique necessary context was identified in this calibration pass.
CASES = {
    "pilot_001": ("negative", "none", None, None, "The import action, source and destination are named in B."),
    "pilot_005": ("positive", "list_intro", 14, "To import a space or a project into an existing space:", "B is a step in the procedure named by the earlier introduction; the adjacent list item is not the required context."),
    "pilot_008": ("negative", "none", None, None, "B explicitly names classification/regression and imputation methods; the adjacent HTML comment is not necessary."),
    "pilot_014": ("positive", "definition_use", 2, "Recency, Frequency, Monetary (RFM)", "B uses RFM, whose expansion is supplied by A."),
    "pilot_025": ("positive", "list_intro", 5, "To create the connection asset, you need these connection details:", "Port number is one of the connection details named by A, not by the adjacent list item."),
    "pilot_027": ("positive", "list_intro", 21, "**API key**: Enter an IAM API key\\. Prerequisites:", "B is presented as a prerequisite for the API key; the adjacent HTML comment contributes no meaning."),
    "pilot_031": ("uncertain", "other", None, None, "The link names sample-size editing, but whether the Data Refinery flow context is necessary for the intended interpretation needs a second reading."),
    "pilot_038": ("negative", "none", None, None, "B names a tutorial and introduces its tasks; the preceding heading is useful context but not necessary for understanding this sentence."),
    "pilot_039": ("negative", "none", None, None, "The task title explicitly says to add a notebook to a project; the preceding list item is not needed."),
    "pilot_049": ("positive", "heading_scope", 1, "Methods for tuning foundation models", "The heading specifies which models the otherwise generic tuning-method introduction concerns."),
    "pilot_050": ("positive", "heading_scope", 1, "Methods for tuning foundation models", "The heading scopes the generic word Models; the adjacent paragraph is not the necessary A."),
    "pilot_052": ("negative", "none", None, None, "B defines prompt-tuning and its effect directly; the fine-tuning note next to it is not needed."),
}


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    queue = {row["record_id"]: row for row in load(PILOT / "queue.jsonl")}
    documents = {row["doc_id"]: row for row in load(PILOT / "documents.jsonl")}
    output = []
    for record_id, (decision, relation, block_number, phrase, rationale) in CASES.items():
        row = queue[record_id]
        document = documents[row["doc_id"]]
        block = next((b for b in document["blocks"] if b["block_id"].endswith(f":{block_number}")), None) if block_number else None
        if block:
            position = block["text"].find(phrase)
            if position < 0:
                raise ValueError(f"A quote not found for {record_id}")
            a_start = block["char_start"] + position
            a_end = a_start + len(phrase)
            a_id, a_span, a_quote = block["block_id"], [a_start, a_end], phrase
        else:
            a_id, a_span, a_quote = "", [], ""
        output.append({"record_id": record_id, "doc_id": row["doc_id"],
                       "b_block_id": row["b_block_id"], "b_span": row["b_span"], "b_quote": row["b_quote"],
                       "a_block_id": a_id, "a_span": a_span, "a_quote": a_quote,
                       "decision": decision, "relation_type": relation,
                       "missing_context": "none identified" if decision == "negative" else rationale,
                       "rationale": rationale, "dependency_group": "",
                       "annotator_id": "codex_ai_draft", "review_status": "ai_draft",
                       "label_provenance": "AI suggestion; no human annotation or independent check"})
    with OUT.open("w", encoding="utf-8", newline="\n") as stream:
        for row in output:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"Wrote {len(output)} AI drafts: {OUT}")


if __name__ == "__main__":
    main()
