"""Validate a human dependency-label JSONL against canonical document blocks.

Input documents JSONL: {doc_id, canonical_text, blocks:[{block_id,char_start,char_end}]}.
This is a read-only audit gate; it never infers or changes labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DECISIONS = {"positive", "negative", "uncertain", "not_applicable"}
RELATIONS = {
    "condition", "list_intro", "explanation_example", "definition_use",
    "caption_table", "anaphora", "exception", "heading_scope",
    "procedure_continuation", "other", "none",
    # Pilot-specific labels retained verbatim; do not silently map them to RC/BI.
    "procedural_prerequisite", "parser_artifact", "list_continuation",
    "scope_and_list_continuation", "definition", "scope",
    "external_reference", "list_continuation_and_external_reference",
}


def rows(path: Path):
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                yield number, json.loads(line)


def validate(documents: Path, labels: Path) -> tuple[int, list[str]]:
    errors: list[str] = []
    docs = {}
    for line, doc in rows(documents):
        doc_id = doc["doc_id"]
        if doc_id in docs:
            errors.append(f"documents:{line}: duplicate doc_id {doc_id}")
        docs[doc_id] = doc
    seen = set()
    count = 0
    for line, label in rows(labels):
        count += 1
        prefix = f"labels:{line}"
        try:
            record_id = label["record_id"]
            edge_id = label.get("edge_id", record_id)
            if edge_id in seen:
                errors.append(f"{prefix}: duplicate edge_id {edge_id}")
            seen.add(edge_id)
            doc = docs[label["doc_id"]]
            blocks = {b["block_id"]: b for b in doc["blocks"]}
            for side in ("a", "b"):
                if side == "a" and label["decision"] != "positive" and not label.get("a_block_id"):
                    continue
                block = blocks[label[f"{side}_block_id"]]
                start, end = label[f"{side}_span"]
                if not (isinstance(start, int) and isinstance(end, int)
                        and block["char_start"] <= start < end <= block["char_end"]):
                    errors.append(f"{prefix}: {side}_span outside block")
                elif doc["canonical_text"][start:end] != label[f"{side}_quote"]:
                    errors.append(f"{prefix}: {side}_quote differs from canonical slice")
            if label["decision"] not in DECISIONS:
                errors.append(f"{prefix}: invalid decision")
            if label["relation_type"] not in RELATIONS:
                errors.append(f"{prefix}: invalid relation_type")
            if label["decision"] == "positive" and not label.get("missing_context", "").strip():
                errors.append(f"{prefix}: positive requires missing_context")
            for field in ("annotator_id", "rationale", "review_status"):
                if not str(label.get(field, "")).strip():
                    errors.append(f"{prefix}: missing {field}")
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"{prefix}: invalid record: {exc}")
    return count, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    args = parser.parse_args()
    count, errors = validate(args.documents, args.labels)
    print(json.dumps({"records": count, "valid": not errors, "errors": errors}, ensure_ascii=False))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
