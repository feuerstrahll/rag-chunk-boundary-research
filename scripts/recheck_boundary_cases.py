"""Recalculate four draft boundary cases after any human edits to the workbook.

This checks contiguous canonical spans and constructs a whole-block partition
witness for pilot_047. It does not revise labels or implement a new chunker.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from transformers import AutoTokenizer

IDS = ("pilot_031", "pilot_032", "pilot_033", "pilot_047")


def count(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False, truncation=False, verbose=False)["input_ids"])


def partition_witness(doc: dict, tokenizer, max_tokens: int, forced: tuple[int, int]) -> dict:
    """Construct an ordered partition, splitting oversized non-target blocks at sentences."""
    blocks = doc["blocks"]
    left, right = forced
    chunks = []
    sentence_splits = []

    def split_oversized(index: int) -> None:
        block = blocks[index]
        text = block["text"]
        candidates = sorted({match.end() for match in re.finditer(r"(?<=[.!?])\s+", text)} | {len(text)})
        cursor = 0
        while cursor < len(text):
            fitting = [end for end in candidates if end > cursor and
                       count(tokenizer, text[cursor:end]) <= max_tokens]
            if not fitting:
                raise ValueError(f"block {index + 1} has no fitting sentence boundary after {cursor}")
            end = max(fitting)
            chunks.append({"first_block": index + 1, "last_block": index + 1,
                           "char_start": block["char_start"] + cursor,
                           "char_end": block["char_start"] + end,
                           "tokens": count(tokenizer, text[cursor:end]), "sentence_fragment": True})
            sentence_splits.append(index + 1)
            cursor = end

    def emit(lo: int, hi: int) -> None:
        text = doc["canonical_text"][blocks[lo]["char_start"]:blocks[hi]["char_end"]]
        size = count(tokenizer, text)
        if size > max_tokens:
            if lo == hi and not (left <= lo <= right):
                split_oversized(lo)
                return
            raise ValueError(f"whole-block chunk {lo + 1}..{hi + 1} exceeds budget: {size}")
        chunks.append({"first_block": lo + 1, "last_block": hi + 1,
                       "char_start": blocks[lo]["char_start"], "char_end": blocks[hi]["char_end"],
                       "tokens": size})

    i = 0
    try:
        while i < len(blocks):
            if i == left:
                emit(left, right)
                i = right + 1
                continue
            boundary = left if i < left else len(blocks)
            j = i
            while j + 1 < boundary:
                following = blocks[j + 1]
                if following["type"] == "heading" or following["section_path"] != blocks[i]["section_path"]:
                    break
                text = doc["canonical_text"][blocks[i]["char_start"]:following["char_end"]]
                if count(tokenizer, text) > max_tokens:
                    break
                j += 1
            emit(i, j)
            i = j + 1
    except ValueError as exc:
        return {"exists": False, "reason": str(exc), "chunks_before_failure": chunks}
    if chunks[0]["char_start"] != blocks[0]["char_start"] or chunks[-1]["char_end"] != blocks[-1]["char_end"]:
        raise AssertionError("Witness does not reach document ends")
    if any(chunks[i]["char_end"] > chunks[i + 1]["char_start"] for i in range(len(chunks) - 1)):
        raise AssertionError("Witness overlaps or reorders source text")
    for block in blocks:
        pieces = sorted((max(block["char_start"], chunk["char_start"]),
                         min(block["char_end"], chunk["char_end"])) for chunk in chunks
                        if chunk["char_start"] < block["char_end"] and chunk["char_end"] > block["char_start"])
        if not pieces or pieces[0][0] != block["char_start"] or pieces[-1][1] != block["char_end"] or any(
            pieces[i][1] != pieces[i + 1][0] for i in range(len(pieces) - 1)
        ):
            raise AssertionError(f"Block coverage gap in witness: {block['block_id']}")
    return {"exists": True, "chunks": chunks, "chunk_count": len(chunks),
            "sentence_split_blocks": sorted(set(sentence_splits)),
            "scope": "ordered partition with sentence splits for oversized non-target blocks and no mandatory overlap; geometry witness, not retrieval result"}


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--workbook", type=Path, required=True)
    cli.add_argument("--tokenizer", type=Path, required=True)
    cli.add_argument("--max-tokens", type=int, required=True)
    cli.add_argument("--out", type=Path, required=True)
    args = cli.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    workbook = json.loads(args.workbook.read_text(encoding="utf-8"))
    docs = workbook["documents"]
    records = {row["record_id"]: row for row in workbook["records"]}
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, use_fast=True)
    details = []
    for record_id in IDS:
        record = records[record_id]
        doc = docs[record["doc_id"]]
        blocks = doc["blocks"]
        by_id = {block["block_id"]: (index, block) for index, block in enumerate(blocks)}
        answer = record["answer"]
        anchors = answer["necessary_a"]
        if answer["decision"] != "positive" or not anchors:
            details.append({"record_id": record_id, "status": "no_positive_group_after_review"})
            continue
        positions = [by_id[record["b"]["block_id"]][0], *(by_id[a["block_id"]][0] for a in anchors)]
        lo, hi = min(positions), max(positions)
        minimal_start = min(record["b"]["span"][0], *(a["span"][0] for a in anchors))
        minimal_end = max(record["b"]["span"][1], *(a["span"][1] for a in anchors))
        full_start, full_end = blocks[lo]["char_start"], blocks[hi]["char_end"]
        full_tokens = count(tokenizer, doc["canonical_text"][full_start:full_end])
        minimal_tokens = count(tokenizer, doc["canonical_text"][minimal_start:minimal_end])
        sections = {tuple(blocks[index]["section_path"]) for index in range(lo, hi + 1)}
        item = {"record_id": record_id, "decision": answer["decision"], "review_status": answer["review_status"],
                "a_count": len(anchors), "min_span_interval": [minimal_start, minimal_end],
                "min_span_contiguous_tokens": minimal_tokens,
                "whole_block_interval": [full_start, full_end], "whole_block_tokens": full_tokens,
                "intermediate_text_included": True,
                "same_section_current_policy": len(sections) == 1,
                "block_distance": hi - lo,
                "whole_block_budget_ok": full_tokens <= args.max_tokens,
                "sentence_split_observation": "Splitting intermediate blocks without omitting or duplicating text cannot reduce the token count of the same contiguous canonical interval; actual sentence-boundary availability is a separate check.",
                "parser_issue": answer.get("parser_issue", ""),
                "anchors": anchors}
        if record_id in {"pilot_031", "pilot_032", "pilot_033"}:
            b_start = record["b"]["span"][0]
            alternatives = []
            for index in range(max(0, by_id[record["b"]["block_id"]][0] - 6),
                               by_id[record["b"]["block_id"]][0]):
                block = blocks[index]
                phrase = "Data Refinery flow"
                position = block["text"].find(phrase)
                if position < 0:
                    continue
                a_start = block["char_start"] + position
                start = min(a_start, b_start)
                end = max(a_start + len(phrase), record["b"]["span"][1])
                alternatives.append({"block_id": block["block_id"], "candidate_a_span": [a_start, a_start + len(phrase)],
                                     "contiguous_tokens_to_b": count(tokenizer, doc["canonical_text"][start:end]),
                                     "whole_block_tokens_to_b": count(tokenizer, doc["canonical_text"][
                                         block["char_start"]:blocks[by_id[record["b"]["block_id"]][0]]["char_end"]]),
                                     "semantic_status": "candidate_only_human_must_judge_sufficiency"})
            item["nearby_alternative_mentions"] = alternatives
        if record_id == "pilot_047":
            item["document_partition_witness"] = partition_witness(doc, tokenizer, args.max_tokens, (lo, hi))
        details.append(item)
    result = {"status": "draft_recheck_or_pending_human_review", "max_tokens": args.max_tokens,
              "source_workbook": str(args.workbook), "cases": details,
              "human_verification": all(item.get("review_status") == "adjudicated" for item in details),
              "go_no_go": "not_assessed"}
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"case_count": len(details), "witness_047": details[-1].get("document_partition_witness", {}).get("exists")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
