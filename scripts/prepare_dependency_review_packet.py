"""Create a blind human review packet for four breaks and six controls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

TARGETS = ("pilot_031", "pilot_032", "pilot_033", "pilot_047")
CONTROLS = ("pilot_001", "pilot_005", "pilot_008", "pilot_014", "pilot_025", "pilot_039")


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--workbook", type=Path, required=True)
    cli.add_argument("--out", type=Path, required=True)
    args = cli.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    source = json.loads(args.workbook.read_text(encoding="utf-8"))
    selected = set(TARGETS + CONTROLS)
    records = []
    for record in source["records"]:
        if record["record_id"] not in selected:
            continue
        records.append({"record_id": record["record_id"], "stratum": record["stratum"],
                        "doc_id": record["doc_id"], "b": record["b"],
                        "answer": {"missing_context": "", "decision": "", "relation_type": "",
                                   "necessary_a": [], "dependency_group": "", "rationale": "",
                                   "parser_issue": "", "annotator_id": "", "review_status": "draft"}})
    if len(records) != len(selected):
        raise ValueError("Review packet is missing a record")
    document_ids = {record["doc_id"] for record in records}
    packet = {"schema_version": "1.0", "status": "blind_human_review_pending",
              "selection_rule": "four draft split cases plus three preserved random positives, two random negatives, and one diagnostic negative; IDs fixed before human review",
              "record_ids": [record["record_id"] for record in records],
              "records": records,
              "documents": {doc_id: source["documents"][doc_id] for doc_id in sorted(document_ids)},
              "instructions": "Read B first; then inspect document blocks and source_markdown. Do not look at AI answers or geometry report until your independent decision is saved. Retain uncertain and parser exclusions."}
    args.out.write_text(json.dumps(packet, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {len(records)} blind records")


if __name__ == "__main__":
    main()
