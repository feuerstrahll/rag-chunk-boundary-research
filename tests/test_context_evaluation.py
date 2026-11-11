from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from article_research.context_evaluation import (
    contains,
    enumerate_heading_links,
    index_preserved,
    retrieved_state,
)


def chunk(a, b, doc="d"):
    return {"doc_id": doc, "char_start": a, "char_end": b}


def test_joint_a_and_b_require_same_index_chunk_but_only_union_after_search():
    group = {"doc_id": "d", "a_spans": [[0, 5], [10, 15]], "b_span": [30, 40]}
    separate = [chunk(0, 20), chunk(25, 40)]
    assert not index_preserved(group, separate)
    assert retrieved_state(group, separate)["complete"]
    assert index_preserved(group, [chunk(0, 40)])
    assert not retrieved_state(group, [chunk(10, 40)])["complete"]


def test_union_detects_real_gap_and_respects_document_identity():
    assert contains([0, 10], [chunk(0, 6), chunk(4, 10)])
    assert not contains([0, 10], [chunk(0, 4), chunk(5, 10)])
    assert contains([0, 10], [chunk(0, 4), chunk(4, 10)])
    group = {"doc_id": "d", "a_spans": [[0, 5]], "b_span": [5, 10]}
    assert not retrieved_state(group, [chunk(0, 10, "other")])["b_fully_returned"]


def test_partial_b_and_no_chunks_are_not_counted_as_complete():
    group = {"doc_id": "d", "a_spans": [[0, 5]], "b_span": [5, 10]}
    assert not retrieved_state(group, [chunk(0, 9)])["complete"]
    assert not retrieved_state(group, [])["complete"]


def test_independent_subtree_enumerator_skips_comments_and_expires_empty_siblings_and_roots():
    entries = [(1, "empty root"), (1, "root"), (2, "empty child"), (2, "child"),
               (None, "<!-- comment -->"), (3, "deep"), (None, "body")]
    blocks = [SimpleNamespace(type="heading" if level else "paragraph", level=level, text=text,
                              block_id=f"b{i}", char_start=i * 40, char_end=i * 40 + len(text))
              for i, (level, text) in enumerate(entries)]
    links = enumerate_heading_links(SimpleNamespace(blocks=blocks))
    assert {r["link_id"] for r in links} == {"b1", "b3", "b5"}
    assert {r["b_block_id"] for r in links} == {"b6"}
