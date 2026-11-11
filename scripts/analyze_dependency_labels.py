"""Summarize verified human labels; never infer labels or a go/no-go decision."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from validate_dependency_labels import rows, validate


def contains(chunk: dict, spans: list[list[int]]) -> bool:
    return all(chunk["char_start"] <= start and end <= chunk["char_end"] for start, end in spans)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--chunks", type=Path, help="JSONL with doc_id,char_start,char_end; optional")
    parser.add_argument("--tokenizer-path", type=Path, help="Pinned local tokenizer snapshot")
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--window-blocks", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    count, errors = validate(args.documents, args.labels)
    if errors:
        raise ValueError("Invalid labels: " + "; ".join(errors))
    labels = [item for _, item in rows(args.labels)]
    if any(item["review_status"] != "adjudicated" for item in labels):
        raise ValueError("All labels must be adjudicated before analysis")
    chunks = defaultdict(list)
    documents = {item["doc_id"]: item for _, item in rows(args.documents)}
    tokenizer = None
    if args.tokenizer_path:
        if not args.max_tokens or args.max_tokens <= 0:
            raise ValueError("--max-tokens must be positive with --tokenizer-path")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, local_files_only=True, use_fast=True)
    if args.chunks:
        for _, chunk in rows(args.chunks):
            chunks[chunk["doc_id"]].append(chunk)
    decisions = Counter(item["decision"] for item in labels)
    types = Counter(item["relation_type"] for item in labels if item["decision"] == "positive")
    preservation = Counter()
    obstacles = Counter()
    simple_rules = Counter()
    local_pattern = Counter()
    external_anchors = Counter()
    repairability = Counter()
    groups = defaultdict(list)
    for item in labels:
        if item["decision"] != "positive":
            continue
        spans = [item["a_span"], item["b_span"]]
        if args.chunks:
            preservation["preserved" if any(contains(chunk, spans) for chunk in chunks[item["doc_id"]]) else "split"] += 1
        else:
            preservation["not_evaluated_no_chunks"] += 1
        obstacles.update(item.get("obstacles", []))
        if tokenizer:
            doc = documents[item["doc_id"]]
            by_id = {block["block_id"]: (i, block) for i, block in enumerate(doc["blocks"])}
            ai, ab = by_id[item["a_block_id"]]
            bi, bb = by_id[item["b_block_id"]]
            start, end = min(item["a_span"][0], item["b_span"][0]), max(item["a_span"][1], item["b_span"][1])
            reasons = []
            if len(tokenizer(doc["canonical_text"][start:end], add_special_tokens=False, truncation=False)["input_ids"]) > args.max_tokens:
                reasons.append("token_limit")
            if ab["section_path"] != bb["section_path"]:
                reasons.append("section_boundary_current_policy")
            if abs(ai - bi) > args.window_blocks:
                reasons.append("window_current_policy")
            if "parser_error" in item.get("obstacles", []):
                reasons.append("parser_error")
            repairability["local_boundary_candidate" if not reasons else "blocked_under_current_policy"] += 1
            obstacles.update(reasons)
        else:
            repairability["not_evaluated_no_tokenizer"] += 1
        simple_rules[str(item.get("simple_structural_rule", "unreviewed"))] += 1
        local_pattern[str(item.get("local_pattern", "unreviewed"))] += 1
        external_anchors[str(item.get("external_anchor", "unreviewed"))] += 1
        if item.get("dependency_group"):
            groups[(item["doc_id"], item["dependency_group"])].append(item)
    group_preservation = Counter()
    for (doc_id, _), members in groups.items():
        spans = [members[0]["b_span"], *(member["a_span"] for member in members)]
        if args.chunks:
            group_preservation["preserved" if any(contains(chunk, spans) for chunk in chunks[doc_id]) else "split"] += 1
        else:
            group_preservation["not_evaluated_no_chunks"] += 1
    result = {"status": "descriptive_human_audit", "edge_count": count,
                  "reviewed_record_count": len({item["record_id"] for item in labels}), "decisions": dict(decisions),
                  "positive_relation_types": dict(types), "pair_preservation": dict(preservation),
                  "group_preservation": dict(group_preservation), "obstacles": dict(obstacles),
                  "simple_structural_rule": dict(simple_rules), "local_pattern": dict(local_pattern),
                  "literature_rc_bi_increment": "not_evaluated_requires_source_definitions_and_separate_annotation",
                  "external_anchor": dict(external_anchors),
                  "locally_repairable_boundary": dict(repairability),
                  "repairability_scope": "necessary feasibility screen under specified local window and current section policy; not proof of retrieval gain or principled impossibility"}
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
