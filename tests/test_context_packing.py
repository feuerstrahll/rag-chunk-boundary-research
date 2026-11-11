"""Behavioral checks for context intervals; no corpus/model downloads."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ARTICLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ARTICLE / "src"))
sys.path.insert(0, str(ARTICLE / "data/intermediate/upstream_snapshot/7f2f09d07dbd925fcdcdc6e6c5b564d355fc287aa1e5eccb7da9633648a9cd89/src"))

from article_research.context_packing import (
    ContextAwarePackingChunker,
    ContextPackingConfig,
    intro_links,
    structural_links,
)
from article_research.hierarchy_packing import (
    HierarchyAwarePackingChunker,
    HierarchyPackingConfig,
)
from chunking.fixed import TokenizerConfig
from models import Block, Document
from test_hierarchy_packing import Words, doc, spans


def pack(document, *, headings=True, intro=False, target=30, hard=40, overlap=4):
    config = ContextPackingConfig(tokenizer=TokenizerConfig(name="fixture", revision="local-test-tokenizer"),
                                  target_tokens=target, max_tokens=hard, overlap_tokens=overlap,
                                  heading_chains=headings, intro_rule=intro)
    chunker = ContextAwarePackingChunker(config, tokenizer=Words())
    chunks = chunker.chunk(document)
    assert all(c.text == document.canonical_text[c.char_start:c.char_end] and c.token_count <= hard for c in chunks)
    assert all(any(c.char_start <= m.start() < c.char_end for c in chunks)
               for m in re.finditer(r"\S", document.canonical_text))
    return chunker, chunks


def test_nested_chain_including_comments_crosses_internal_parent_boundary():
    d = doc([(1, "root"), (None, "<!-- metadata -->"), (2, "child"), (3, "grandchild"),
             (None, "<!-- <ul> -->"), (None, "actual body")])
    chunker, chunks = pack(d)
    assert len(chunks) == 1
    assert chunks[0].text == d.canonical_text
    assert len(chunker.links) == 3
    assert all(link["preserved_in_index"] for link in chunker.links)


def test_chain_moves_to_next_chunk_before_target_and_no_overlap_across_parent():
    d = doc([(1, "root"), (None, "old body words"), (2, "section"), (3, "subsection"), (None, "new body")])
    _, chunks = pack(d, target=6, hard=10, overlap=2)
    assert any(c.text == "section\n\nsubsection\n\nnew body" for c in chunks)
    assert not any(c.text.endswith("subsection") or c.text.endswith("section") for c in chunks)


def test_empty_sibling_does_not_claim_other_siblings_body_but_pending_ancestor_can():
    d = doc([(1, "root"), (2, "empty"), (2, "sibling"), (None, "body")])
    links, empty = structural_links(d)
    assert {r["link_id"] for r in links} == {"b0", "b2"}
    assert empty[0]["block_id"] == "b1"
    assert len(pack(d)[1]) == 1


def test_empty_root_is_never_attached_to_next_root_even_same_title():
    d = doc([(1, "same"), (None, "<!-- no body -->"), (1, "same"), (2, "child"), (None, "body")])
    chunker, chunks = pack(d)
    assert all(r["link_id"] != "b0" for r in chunker.links)
    assert chunks[0].char_end <= d.blocks[2].char_start
    assert any(r["reason"] == "empty_subtree_before_next_heading" for r in chunker.diagnostics)


@pytest.mark.parametrize("size,expected", [(480, True), (481, True), (510, True), (511, False)])
def test_exact_encoder_edges_preferred_chain(size, expected):
    d = doc([(1, "root"), (2, "child"), (None, " ".join(["body"] * (size - 2)))])
    chunker, chunks = pack(d, target=480, hard=510, overlap=64)
    assert chunker.links[0]["preserved_in_index"] is expected
    assert all(c.token_count <= 510 for c in chunks)
    if not expected:
        assert any(r["reason"] == "preferred_unit_exceeds_encoder_inherited_token_fallback" for r in chunker.diagnostics)
        assert chunker.links[1]["preserved_in_index"]  # feasible descendant survives an oversized ancestor


def connection(items, intro="To create the connection asset, you need these connection details:"):
    entries = [(1, "IBM Watson Query connection"), (2, "Create a Watson Query connection"),
               (None, intro), (None, "<!-- <ul> -->")]
    entries.extend((None, item) for item in items)
    d = doc(entries)
    for b in d.blocks[4:]:
        b.type = "list"
    return d


def test_port_requires_intro_and_concrete_heading_direction_is_b_to_a():
    d = connection(["- **Database name**", "- **Port number**", "- **Instance ID**"])
    links, candidates = intro_links(d)
    assert len(links) == 3
    assert all(r["a_block_ids"] == ["b1", "b2"] for r in links)
    assert all(r["b_block_id"] in {"b4", "b5", "b6"} for r in links)
    assert all(r["decision"] == "propose" for r in candidates)
    chunker, chunks = pack(d, intro=True, target=25, hard=40)
    assert all(link["preserved_in_index"] for link in chunker.links)
    assert any("<!-- <ul> -->" in c.text and "Port number" in c.text for c in chunks)


def test_long_list_does_not_destroy_shorter_feasible_item_links():
    d = connection(["- Database name", "- Port number", *["- Instance ID"] * 20])
    chunker, chunks = pack(d, intro=True, target=25, hard=30)
    links = [r for r in chunker.links if r["kind"] == "intro"]
    assert any(r["preserved_in_index"] for r in links)
    assert any(r["packing_decision"] == "token_limit_keep_legal_base_units" for r in links)
    assert all(c.token_count <= 30 for c in chunks)


def test_rule_abstains_on_self_contained_item_and_unsupported_anaphora():
    d = connection(["- Set the Watson Query connection port to 50000.", "- This setting is recommended.",
                    "- In this case, use port 80.", "- [Port number](https://example.test)"])
    assert not intro_links(d)[0]
    assert {r["decision"] for r in intro_links(d)[1]} == {"abstain"}


def test_ambiguous_generic_object_and_heading_transition_end_scope():
    d = connection(["- Port number"])
    d = Document(doc_id="d", source="fixture", blocks=[Block(block_id=b.block_id, type=b.type, text=b.text,
                 level=b.level) if i != 1 else Block(block_id=b.block_id, type=b.type,
                 text="Configuration", level=b.level) for i, b in enumerate(d.blocks)])
    assert not intro_links(d)[0]
    d = connection(["- Port number"])
    entries = [Block(block_id=b.block_id, type=b.type, text=b.text, level=b.level) for b in d.blocks]
    entries.insert(4, Block(block_id="new", type="heading", text="Other", level=2))
    d = Document(doc_id="d", source="fixture", blocks=entries)
    assert not intro_links(d)[0]


def test_two_flags_disabled_match_v1_spans_and_v1_config_hash_is_untouched():
    d = doc([(1, "root"), (2, "child"), (None, "<!-- comment -->"), (3, "deep"), (None, "body")])
    base = HierarchyPackingConfig(tokenizer=TokenizerConfig(name="fixture", revision="local-test-tokenizer"),
                                   target_tokens=30, max_tokens=40, overlap_tokens=4)
    v1 = HierarchyAwarePackingChunker(base, tokenizer=Words())
    off = pack(d, headings=False)[1]
    assert spans(off) == spans(v1.chunk(d))
    assert [c.chunk_id for c in off] == [c.chunk_id for c in v1.chunk(d)]
    assert v1.provenance["algorithm_version"] == "1"


def test_intro_toggle_changes_only_selected_context_units():
    d = connection(["- Database name", "- Port number", "- Instance ID"])
    off, _ = pack(d, target=22, hard=40)
    on, _ = pack(d, intro=True, target=22, hard=40)
    assert not off.candidates
    assert len([r for r in on.links if r["kind"] == "intro"]) == 3
    assert off.provenance["intro_rule"] is False and on.provenance["intro_rule"] is True


def test_equal_spans_keep_chunk_ids_and_retrieval_ties_across_flags():
    d = doc([(1, "root"), (None, "fully independent text")])
    _, off = pack(d)
    _, on = pack(d, intro=True)
    assert [(c.chunk_id, c.text) for c in off] == [(c.chunk_id, c.text) for c in on]
