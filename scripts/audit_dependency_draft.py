"""Exploratory geometry audit of draft dependency labels; no go/no-go inference."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer

from scripts.validate_dependency_labels import rows, validate


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if args.max_tokens <= 0:
        raise ValueError("max-tokens must be positive")
    _, errors = validate(args.documents, args.labels)
    if errors:
        raise ValueError("; ".join(errors))
    documents = {doc["doc_id"]: doc for _, doc in rows(args.documents)}
    labels = [label for _, label in rows(args.labels)]
    by_record = defaultdict(list)
    for label in labels:
        by_record[label["record_id"]].append(label)
    chunk_rows = pq.read_table(args.chunks, columns=["doc_id", "chunk_id", "char_start", "char_end", "text"]).to_pylist()
    chunks = defaultdict(list)
    for chunk in chunk_rows:
        if chunk["doc_id"] in documents:
            doc = documents[chunk["doc_id"]]
            start, end = chunk["char_start"], chunk["char_end"]
            if doc["canonical_text"][start:end] != chunk["text"]:
                raise ValueError(f"Baseline chunk/canonical mismatch: {chunk['chunk_id']}")
            chunks[chunk["doc_id"]].append(chunk)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, use_fast=True)
    cases = []
    for record_id, edges in by_record.items():
        first = edges[0]
        decision = first["decision"]
        if any(edge["decision"] != decision or edge["doc_id"] != first["doc_id"] for edge in edges):
            raise ValueError(f"Inconsistent edges: {record_id}")
        case = {"record_id": record_id, "stratum": first["stratum"], "decision": decision,
                "relation_type": first["relation_type"], "edge_count": len(edges),
                "review_status": first["review_status"], "parser_issue": first.get("parser_issue", ""),
                "simple_structural_rule_draft": first.get("simple_structural_rule", ""),
                "external_anchor_draft": first.get("external_anchor", "")}
        if decision == "positive":
            doc = documents[first["doc_id"]]
            spans = [first["b_span"], *(edge["a_span"] for edge in edges)]
            containing = [chunk["chunk_id"] for chunk in chunks[first["doc_id"]]
                          if all(chunk["char_start"] <= start and end <= chunk["char_end"] for start, end in spans)]
            blocks = {block["block_id"]: (i, block) for i, block in enumerate(doc["blocks"])}
            block_positions = [blocks[first["b_block_id"]][0], *(blocks[edge["a_block_id"]][0] for edge in edges)]
            left, right = min(block_positions), max(block_positions)
            full_start = doc["blocks"][left]["char_start"]
            full_end = doc["blocks"][right]["char_end"]
            token_count = len(tokenizer(doc["canonical_text"][full_start:full_end],
                                        add_special_tokens=False, truncation=False)["input_ids"])
            sections = {tuple(doc["blocks"][i]["section_path"]) for i in range(left, right + 1)}
            case.update(preserved=bool(containing), containing_chunk_ids=containing,
                        contiguous_whole_block_tokens=token_count,
                        token_limit_ok=token_count <= args.max_tokens,
                        same_section_current_policy=len(sections) == 1,
                        block_distance=right - left,
                        local_window_3_ok=right - left <= 3,
                        feasible_under_screen=(token_count <= args.max_tokens and len(sections) == 1 and right - left <= 3),
                        screen_scope="necessary local feasibility only; not a constructed chunk or retrieval result")
        cases.append(case)
    counts = Counter((case["stratum"], case["decision"]) for case in cases)
    positives = [case for case in cases if case["decision"] == "positive"]
    report = {
        "status": "exploratory_ai_draft_geometry_audit", "go_no_go": "not_assessed",
        "label_provenance": "AI_ChatGPT draft; human verification pending",
        "inputs_sha256": {"documents": sha256(args.documents), "labels": sha256(args.labels),
                          "chunks": sha256(args.chunks)},
        "max_tokens": args.max_tokens, "tokenizer_path": str(args.tokenizer),
        "record_count": len(cases), "edge_count": len(labels),
        "stratum_decision_counts": [{"stratum": stratum, "decision": decision, "count": count}
                                    for (stratum, decision), count in sorted(counts.items())],
        "positive_preserved": sum(case["preserved"] for case in positives),
        "positive_split": sum(not case["preserved"] for case in positives),
        "split_feasible_under_screen": sum(not case["preserved"] and case["feasible_under_screen"] for case in positives),
        "cases": cases,
    }
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("record_count", "edge_count", "positive_preserved",
                                                   "positive_split", "split_feasible_under_screen")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
