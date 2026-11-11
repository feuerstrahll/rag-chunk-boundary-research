"""Separate one-chunk index retention from union-of-retrieved-chunks context.

This module never infers semantic labels from detector proposals.
"""
from __future__ import annotations

import re


def enumerate_heading_links(document):
    """Independent per-subtree scan, without the chunker's pending stack.

    The operational content definition is the same block-local comment exclusion;
    enumeration and preservation checks do not consume detector predictions.
    """
    links = []
    for index, heading in enumerate(document.blocks):
        if heading.type != "heading" or not heading.text:
            continue
        for body in document.blocks[index + 1:]:
            if body.type == "heading":
                if body.level <= heading.level:
                    break
                continue
            if not re.sub(r"<!--(?:.*?-->|.*$)", "", body.text, flags=re.DOTALL).strip():
                continue
            links.append({"kind": "heading", "link_id": heading.block_id,
                          "a_spans": [[heading.char_start, heading.char_end]], "a_block_ids": [heading.block_id],
                          "b_span": [body.char_start, body.char_end], "b_block_id": body.block_id,
                          "start": heading.char_start, "end": body.char_end,
                          "label_provenance": "independent_subtree_scan_structural_definition_not_semantic_label"})
            break
    return links


def contains(span, chunks):
    """Exact character coverage by a union, with no double counting of overlap."""
    left, right = span
    cursor = left
    for chunk in sorted(chunks, key=lambda c: (c["char_start"], c["char_end"])):
        start, end = chunk["char_start"], chunk["char_end"]
        if end <= cursor:
            continue
        if start > cursor:
            break
        cursor = end
        if cursor >= right:
            return True
    return False


def index_preserved(group, chunks):
    required = [group["b_span"], *group["a_spans"]]
    return any(all(c["char_start"] <= a and c["char_end"] >= b for a, b in required)
               for c in chunks if c["doc_id"] == group["doc_id"])


def retrieved_state(group, chunks):
    local = [c for c in chunks if c["doc_id"] == group["doc_id"]]
    b_returned = contains(group["b_span"], local)
    a_returned = all(contains(span, local) for span in group["a_spans"])
    return {"b_fully_returned": b_returned, "all_a_fully_returned": a_returned,
            "complete": b_returned and a_returned}
