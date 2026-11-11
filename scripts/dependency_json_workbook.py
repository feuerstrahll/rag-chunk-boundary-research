"""Create an editable JSON workbook and export its completed labels to JSONL."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.validate_dependency_labels import DECISIONS, RELATIONS, validate

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "pilots" / "dependency_pilot_20261022_v8"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def create(path: Path) -> None:
    if path.exists():
        raise FileExistsError(path)
    documents = read_jsonl(SOURCE / "documents.jsonl")
    queue = read_jsonl(SOURCE / "queue.jsonl")
    by_doc = {doc["doc_id"]: doc for doc in documents}
    records = []
    for item in queue:
        blocks = by_doc[item["doc_id"]]["blocks"]
        index = next(i for i, block in enumerate(blocks) if block["block_id"] == item["b_block_id"])
        records.append({
            "record_id": item["record_id"], "stratum": item["stratum"], "doc_id": item["doc_id"],
            "b": {"block_id": item["b_block_id"], "span": item["b_span"], "quote": item["b_quote"]},
            "nearby_blocks": [{key: block[key] for key in ("block_id", "type", "char_start", "char_end", "text")}
                              for block in blocks[max(0, index - 2): index + 3]],
            "answer": {"missing_context": "", "decision": "", "relation_type": "",
                       "necessary_a": [], "dependency_group": "", "rationale": "",
                       "parser_issue": "", "simple_structural_rule": "", "local_pattern": "",
                       "external_anchor": "", "annotator_id": "", "review_status": "draft"},
        })
    workbook = {
        "schema_version": "1.0", "status": "unlabeled_json_workbook",
        "source_manifest": str(SOURCE / "manifest.json"),
        "instructions": [
            "Read b alone and fill missing_context before using nearby_blocks or documents.",
            "Open documents[doc_id].blocks and source_markdown when nearby context is insufficient.",
            "For positive, necessary_a is a list of one or more {block_id, span:[start,end]} objects; use minimal canonical spans.",
            "For uncertain, necessary_a may hold partially identified candidate spans; these are not counted as positive dependencies.",
            "Use negative only when B is self-contained, not merely when an adjacent A is unnecessary.",
            "Use uncertain if needed context cannot be uniquely located; use not_applicable for parser/coordinate defects.",
            "Keep review_status=draft until an actual independent check and adjudication.",
        ],
        "documents": {doc["doc_id"]: doc for doc in documents}, "records": records,
    }
    path.write_text(json.dumps(workbook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Created {len(records)} unlabeled records: {path}")


def export(workbook_path: Path, output_path: Path, *, allow_incomplete: bool) -> None:
    if output_path.exists():
        raise FileExistsError(output_path)
    workbook = json.loads(workbook_path.read_text(encoding="utf-8"))
    docs = workbook["documents"]
    records = workbook["records"]
    if len(records) != 54 or len({row["record_id"] for row in records}) != 54:
        raise ValueError("Workbook must retain all 54 unique records")
    labels = []
    incomplete = []
    for row in records:
        answer = row["answer"]
        decision = answer["decision"]
        if not decision:
            incomplete.append(row["record_id"])
            continue
        if decision not in DECISIONS:
            raise ValueError(f"{row['record_id']}: invalid decision")
        if answer["relation_type"] not in RELATIONS:
            raise ValueError(f"{row['record_id']}: invalid relation_type")
        if not answer["annotator_id"].strip() or not answer["rationale"].strip():
            raise ValueError(f"{row['record_id']}: annotator_id and rationale required")
        anchors = answer["necessary_a"]
        if decision == "positive" and not anchors:
            raise ValueError(f"{row['record_id']}: positive needs at least one A")
        if decision not in {"positive", "uncertain"} and anchors:
            raise ValueError(f"{row['record_id']}: only positive or uncertain may contain A spans")
        if decision == "positive" and not answer["missing_context"].strip():
            raise ValueError(f"{row['record_id']}: describe missing context")
        doc = docs[row["doc_id"]]
        blocks = {block["block_id"]: block for block in doc["blocks"]}
        if len(anchors) > 1 and not answer["dependency_group"]:
            raise ValueError(f"{row['record_id']}: multiple A require dependency_group")
        for index, anchor in enumerate(anchors or [None], 1):
            if anchor:
                block = blocks[anchor["block_id"]]
                start, end = anchor["span"]
                if not (block["char_start"] <= start < end <= block["char_end"]):
                    raise ValueError(f"{row['record_id']}: A span outside block")
                a_id, a_span, a_quote = block["block_id"], [start, end], doc["canonical_text"][start:end]
            else:
                a_id, a_span, a_quote = "", [], ""
            labels.append({
                "record_id": row["record_id"], "edge_id": f"{row['record_id']}:{index}",
                "doc_id": row["doc_id"], "stratum": row["stratum"],
                "b_block_id": row["b"]["block_id"], "b_span": row["b"]["span"], "b_quote": row["b"]["quote"],
                "a_block_id": a_id, "a_span": a_span, "a_quote": a_quote,
                "decision": decision, "relation_type": answer["relation_type"],
                "a_status": "partial_candidate" if decision == "uncertain" and anchor else "asserted_in_draft" if anchor else "none",
                "missing_context": answer["missing_context"], "rationale": answer["rationale"],
                "dependency_group": answer["dependency_group"], "parser_issue": answer["parser_issue"],
                "simple_structural_rule": answer["simple_structural_rule"],
                "local_pattern": answer["local_pattern"], "external_anchor": answer["external_anchor"],
                "annotator_id": answer["annotator_id"], "review_status": answer["review_status"],
            })
    if incomplete and not allow_incomplete:
        raise ValueError(f"Incomplete records: {', '.join(incomplete)}")
    if not labels:
        raise ValueError("No completed labels to export")
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        temporary.write_text("".join(json.dumps(label, ensure_ascii=False, sort_keys=True) + "\n" for label in labels), encoding="utf-8")
        count, errors = validate(SOURCE / "documents.jsonl", temporary)
        if errors:
            raise ValueError("; ".join(errors))
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Exported {count} edges from {len({label['record_id'] for label in labels})} records: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create_parser = sub.add_parser("create")
    create_parser.add_argument("--output", type=Path, required=True)
    export_parser = sub.add_parser("export")
    export_parser.add_argument("--workbook", type=Path, required=True)
    export_parser.add_argument("--output", type=Path, required=True)
    export_parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    if args.command == "create":
        create(args.output)
    else:
        export(args.workbook, args.output, allow_incomplete=args.allow_incomplete)


if __name__ == "__main__":
    main()
