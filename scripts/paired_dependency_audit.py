"""Compare draft dependency spans against existing structure and selected fixed windows."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def fixed_windows(text: str, tokenizer: object, size: int = 480, overlap: int = 20) -> list[tuple[int, int]]:
    offsets = tokenizer(text, add_special_tokens=False, truncation=False, return_offsets_mapping=True)["offset_mapping"]
    windows = []
    start = 0
    while start < len(offsets):
        end = min(start + size, len(offsets))
        while end > start:
            left, right = offsets[start][0], offsets[end - 1][1]
            if len(tokenizer(text[left:right], add_special_tokens=False, truncation=False)["input_ids"]) <= size:
                break
            end -= 1
        if end == start:
            raise ValueError("One token cannot fit")
        windows.append((left, right))
        if end == len(offsets):
            break
        start = max(start + 1, end - overlap)
    return windows


def preserved(windows: list[tuple[int, int]], spans: list[list[int]]) -> bool:
    return any(all(left <= a and b <= right for a, b in spans) for left, right in windows)


def main() -> None:
    cli = argparse.ArgumentParser()
    for name in ("documents", "labels", "structure_chunks", "tokenizer", "selection", "out"):
        cli.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = cli.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    assert selection["primary"]["candidate_id"] == "fixed_m480_o20"
    docs = {row["doc_id"]: row for row in read_jsonl(args.documents)}
    labels = defaultdict(list)
    for row in read_jsonl(args.labels):
        labels[row["record_id"]].append(row)
    chunks = defaultdict(list)
    for row in pq.read_table(args.structure_chunks, columns=["doc_id", "char_start", "char_end", "text"]).to_pylist():
        if row["doc_id"] in docs:
            assert docs[row["doc_id"]]["canonical_text"][row["char_start"]:row["char_end"]] == row["text"]
            chunks[row["doc_id"]].append((row["char_start"], row["char_end"]))
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, use_fast=True)
    positive_doc_ids = {edges[0]["doc_id"] for edges in labels.values() if edges[0]["decision"] == "positive"}
    fixed = {doc_id: fixed_windows(docs[doc_id]["canonical_text"], tokenizer) for doc_id in sorted(positive_doc_ids)}
    result = []
    for record_id, edges in sorted(labels.items()):
        first = edges[0]
        item = {"record_id": record_id, "stratum": first["stratum"], "decision": first["decision"],
                "review_status": first["review_status"], "edge_count": len(edges)}
        if first["decision"] == "positive":
            spans = [first["b_span"], *(edge["a_span"] for edge in edges)]
            doc_id = first["doc_id"]
            structure_ok = preserved(chunks[doc_id], spans)
            fixed_ok = preserved(fixed[doc_id], spans)
            item.update(structure_preserved=structure_ok, fixed_preserved=fixed_ok,
                        pair_class=("both_preserve" if structure_ok and fixed_ok else
                                    "structure_only" if structure_ok else
                                    "fixed_only" if fixed_ok else "both_split"))
        result.append(item)
    counts = Counter(item["pair_class"] for item in result if item["decision"] == "positive")
    output = {"status": "draft_paired_geometry_audit", "interpretation": "spans only; not retrieval or prevalence",
              "selection": "fixed_m480_o20", "fixed_window_algorithm": "retokenized-v2",
              "inputs_sha256": {name: digest(getattr(args, name)) for name in
                                ("documents", "labels", "structure_chunks", "selection")},
              "positive_counts": dict(counts), "cases": result}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output["positive_counts"]))


if __name__ == "__main__":
    main()
