"""Apply the recorded content decisions without altering the original draft."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

NEGATIVE = {"pilot_010", "pilot_020", "pilot_026", "pilot_035", "pilot_038"}
KEEP_ONE = {"pilot_011", "pilot_050"}
CONFIRMED = {"pilot_005", "pilot_011", "pilot_014", "pilot_025", "pilot_040", "pilot_049", "pilot_050"}


def main() -> None:
    cli = argparse.ArgumentParser()
    cli.add_argument("--draft", type=Path, required=True)
    cli.add_argument("--out", type=Path, required=True)
    args = cli.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    rows = [json.loads(line) for line in args.draft.read_text(encoding="utf-8").splitlines() if line]
    output = []
    seen = set()
    for row in rows:
        rid = row["record_id"]
        if rid in NEGATIVE | KEEP_ONE and rid in seen:
            continue
        seen.add(rid)
        if rid in NEGATIVE:
            row.update(decision="negative", relation_type="none", a_block_id="", a_span=[], a_quote="",
                       a_status="none", dependency_group="", missing_context="No necessary external A.",
                       rationale="Content check: B is interpretable without the proposed list or preceding procedural step.",
                       review_status="content_checked")
        elif rid in KEEP_ONE:
            row.update(dependency_group="", review_status="content_checked",
                       rationale="Only the heading supplies the missing subject or scope; following list items are not individually necessary to understand B.")
        elif rid == "pilot_040":
            row.update(a_span=[1655, 1672], a_quote="create a notebook", review_status="content_checked",
                       rationale="The heading's object resolves 'this tutorial'; the rest of the heading is unnecessary.")
        elif rid in CONFIRMED:
            row["review_status"] = "content_checked"
        output.append(row)
    assert {r["record_id"] for r in output if r["decision"] == "positive" and r["review_status"] == "content_checked"} == CONFIRMED
    args.out.parent.mkdir(parents=True, exist_ok=False)
    args.out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output), encoding="utf-8")
    print(f"{len(output)} edges, {len(seen)} records")


if __name__ == "__main__":
    main()
