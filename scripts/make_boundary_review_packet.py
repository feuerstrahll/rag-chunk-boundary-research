"""Present frozen diagnostic cases without chunk coordinates or method labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    cli = argparse.ArgumentParser()
    cli.add_argument("--selection", type=Path, required=True)
    cli.add_argument("--documents", type=Path, required=True)
    cli.add_argument("--out", type=Path, required=True)
    args = cli.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    docs = {d["doc_id"]: d for d in map(json.loads, args.documents.read_text(encoding="utf-8").splitlines())}
    cases = []
    for number, item in enumerate(selection["selected"], 1):
        doc = docs[item["doc_id"]]
        blocks = doc["blocks"]
        positions = [i for i, block in enumerate(blocks) if block["block_id"] in item["block_ids"]]
        left, right = max(0, min(positions) - 1), min(len(blocks), max(positions) + 2)
        cases.append({"case_id": f"boundary_{number:02d}", "doc_id": item["doc_id"],
                      "blocks": [{"block_id": block["block_id"], "text": block["text"]}
                                 for block in blocks[left:right]],
                      "answer": {"decision": "", "b_block_id": "", "necessary_a": [], "rationale": "", "parser_issue": ""}})
    output = {"status": "content_review_pending", "selection_sha256": selection["inputs_sha256"],
              "instructions": "Read blocks without boundary coordinates. Mark only context necessary to understand B; a prior procedural action alone is insufficient. Use uncertain if source representation prevents a decision.",
              "cases": cases}
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {len(cases)} cases")


if __name__ == "__main__":
    main()
