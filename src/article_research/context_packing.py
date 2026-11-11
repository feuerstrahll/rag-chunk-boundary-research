"""V2: preserve explicit, continuous context intervals before greedy packing.

V1 is imported, never modified. The parser, canonical text, chunk constructor,
token counting and oversized token fallback remain the frozen implementations.
No query, answer, gold span or retrieval result is accepted by this module.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from chunking.fixed import _chunk_id, canonical_text_hash, validate_chunking_document
from models import Block, Document

from article_research.hierarchy_packing import (
    HierarchyAwarePackingChunker,
    HierarchyPackingConfig,
)


def visible(text: str) -> str:
    """Detection view only: never used as encoder input or chunk text."""
    return re.sub(r"<!--(?:.*?-->|.*$)", "", text, flags=re.DOTALL).strip()


def plain(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[\\*_#]", "", visible(text))).strip()


@dataclass(frozen=True)
class ContextPackingConfig(HierarchyPackingConfig):
    heading_chains: bool = True
    intro_rule: bool = False


def structural_links(document: Document) -> tuple[list[dict], list[dict]]:
    """Keep pending headings until the first visible body in THEIR subtree.

    Empty subtrees expire on a same/higher-level heading. Ancestors still
    pending can span an empty child, including every intervening byte.
    """
    pending: list[Block] = []
    links, empty = [], []
    for block in document.blocks:
        if not block.text:
            continue
        if block.type == "heading":
            if block.level is None:
                raise ValueError("Heading level is required")
            expired = [h for h in pending if h.level >= block.level]
            empty.extend({"kind": "heading", "block_id": h.block_id,
                          "reason": "empty_subtree_before_next_heading"} for h in expired)
            pending = [h for h in pending if h.level < block.level]
            pending.append(block)
        elif visible(block.text) and pending:
            for heading in pending:
                links.append({
                    "kind": "heading", "link_id": heading.block_id,
                    "a_spans": [[heading.char_start, heading.char_end]],
                    "a_block_ids": [heading.block_id],
                    "b_span": [block.char_start, block.char_end],
                    "b_block_id": block.block_id,
                    "start": heading.char_start, "end": block.char_end,
                    "label_provenance": "deterministic_structural_definition_not_semantic_label",
                })
            pending = []
    empty.extend({"kind": "heading", "block_id": h.block_id,
                  "reason": "empty_subtree_at_document_end"} for h in pending)
    return links, empty


# Frozen on original-train document examples. These lexical gates deliberately
# cover only short connection-field items, a narrow subtype of intro -> item.
INTRO = re.compile(
    r"^To (?P<operation>create|configure|set up|connect to) (?P<object>[^:;.!?]+?), "
    r"(?:you (?:need|must (?:provide|enter|specify))|(?:provide|enter|specify)) "
    r"(?:these|the following) (?:connection )?(?:details|information|values|settings):$", re.IGNORECASE)
FIELD = re.compile(
    r"^(?:database name|hostname(?: or IP address)?(?: of the database)?|IP address|"
    r"port(?: number)?|instance ID|user ?name|user ID|password|API key|"
    r"SSL certificate(?: \(if required by the database server\))?)\.?$", re.IGNORECASE)
CONNECTION_HEADING = re.compile(r"^(?:Create|Configure|Set up|Connect to) (?:a |an |the )?(.+?) connection$", re.IGNORECASE)
GENERIC_OBJECT = re.compile(r"^(?:the |a |an )?(?:connection(?: asset)?|it|this|these|settings?|values?)$", re.IGNORECASE)


def concrete_connection_name(name: str) -> bool:
    # Named product/technology, rather than "new", "existing", "your", etc.
    return bool(re.search(r"\b[A-Z][A-Za-z0-9]+\b", name)) and name.lower() not in {
        "new", "existing", "your", "this", "that", "database", "data source"}


def intro_links(document: Document) -> tuple[list[dict], list[dict]]:
    """Emit B -> A only, with lexical object/scope evidence; otherwise abstain.

    A must immediately precede a list run (ignoring comment-only blocks).
    The run ends on any heading or visible non-list block. Long instructions,
    links, conditions and anaphora outside this closed subtype are rejected.
    """
    links, candidates = [], []
    headings: list[Block] = []
    blocks = document.blocks
    for index, block in enumerate(blocks):
        if block.type == "heading":
            while headings and headings[-1].level >= block.level:
                headings.pop()
            headings.append(block)
            continue
        text = plain(block.text)
        if block.type != "paragraph" or not re.search(r"\b(?:following|these|this setting|in this case)\b", text, re.IGNORECASE):
            continue
        match = INTRO.fullmatch(text)
        reason = None
        anchors = [block]
        if not match:
            reason = "intro_not_explicit_connection_field_schema"
        elif not re.search(r"\bconnection\b", match["object"], re.IGNORECASE):
            reason = "object_not_in_supported_connection_domain"
        elif GENERIC_OBJECT.fullmatch(match["object"]):
            # Generic A needs ONE unambiguous active operation heading that
            # names a concrete connection. Multiple matching ancestors abstain.
            concrete = [h for h in headings if (m := CONNECTION_HEADING.fullmatch(plain(h.text)))
                        and concrete_connection_name(m[1])]
            if len(concrete) != 1 or concrete[0] is not headings[-1]:
                reason = "generic_object_without_unique_concrete_operation_heading"
            else:
                anchors.insert(0, concrete[0])
        elif not ((named := re.fullmatch(r"(?:a |an )?(.+?) connection", match["object"]))
                  and concrete_connection_name(named[1])):
            reason = "object_not_a_concrete_named_connection"
        j = index + 1
        while j < len(blocks) and blocks[j].type != "heading" and not visible(blocks[j].text):
            j += 1
        if j == len(blocks) or blocks[j].type != "list":
            candidates.append({"a_block_id": block.block_id, "decision": "abstain",
                               "reason": reason or "no_immediately_following_list"})
            continue
        while j < len(blocks):
            item = blocks[j]
            if item.type == "heading" or (visible(item.text) and item.type != "list"):
                break
            if item.type == "list":
                item_text = re.sub(r"^\s*[-+*]\s+|^\s*\d+[.)]\s+", "", plain(item.text))
                rejection = reason
                if rejection is None and not FIELD.fullmatch(item_text):
                    rejection = "item_not_closed_short_underspecified_field_phrase"
                candidates.append({"a_block_id": block.block_id, "b_block_id": item.block_id,
                                   "decision": "abstain" if rejection else "propose",
                                   "reason": rejection or "explicit_operation_object_and_underspecified_field"})
                if rejection is None:
                    links.append({
                        "kind": "intro", "link_id": f"intro:{item.block_id}",
                        "a_spans": [[a.char_start, a.char_end] for a in anchors],
                        "a_block_ids": [a.block_id for a in anchors],
                        "b_span": [item.char_start, item.char_end], "b_block_id": item.block_id,
                        "start": anchors[0].char_start, "end": item.char_end,
                        "label_provenance": "AI_proposed_deterministic_heuristic_not_human_annotation",
                    })
            j += 1
    return links, candidates


@dataclass
class Unit:
    blocks: list[Block]
    first_key: tuple[int, int]
    last_key: tuple[int, int]


class ContextAwarePackingChunker(HierarchyAwarePackingChunker):
    def __init__(self, config: ContextPackingConfig, *, tokenizer: Any) -> None:
        super().__init__(config, tokenizer=tokenizer)
        self.span_identity_config_hash = self.chunker_config_hash
        self.context_config = config
        identity = {**super().provenance, "algorithm_version": "2",
                    "heading_chains": config.heading_chains, "intro_rule": config.intro_rule,
                    "intro_rule_version": "connection_fields_1",
                    "binding_policy": "continuous_units_before_parent_partition"}
        identity.pop("chunker_config_hash", None)
        self.chunker_config_hash = sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        self.links: list[dict] = []
        self.candidates: list[dict] = []

    @property
    def provenance(self) -> dict[str, Any]:
        return {**super().provenance, "algorithm_version": "2",
                "heading_chains": self.context_config.heading_chains,
                "intro_rule": self.context_config.intro_rule,
                "intro_rule_version": "connection_fields_1",
                "span_identity_config_hash": self.span_identity_config_hash,
                "chunk_id_policy": "v1_identity_for_equal_canonical_spans_to_preserve_ties"}

    def _make_chunk(self, document, start, end, *, fallback):
        chunk = super()._make_chunk(document, start, end, fallback=fallback)
        identity = _chunk_id(doc_id=document.doc_id, text_hash=canonical_text_hash(document),
                             config_hash=self.span_identity_config_hash, char_start=start, char_end=end)
        return chunk.model_copy(update={"chunk_id": identity})

    def _keys(self, blocks: list[Block]) -> list[tuple[int, int]]:
        stack, keys = [], []
        for index, block in enumerate(blocks):
            if block.type == "heading":
                while stack and stack[-1][1] >= block.level:
                    stack.pop()
                stack.append((index, block.level))
            key = (stack[0][0], stack[-2][0] if len(stack) > 1 else -1) if stack else (-1, -1)
            keys.append(key if self.packing_config.respect_parent_boundary else (-1, -1))
        return keys

    def _context_units(self, document: Document) -> list[Unit]:
        blocks = [b for b in document.blocks if b.text]
        keys = self._keys(blocks)
        positions = {b.block_id: i for i, b in enumerate(blocks)}
        structural, empty = structural_links(document)
        self.diagnostics.extend(empty)
        semantic, self.candidates = intro_links(document) if self.context_config.intro_rule else ([], [])
        self.links = [*(structural if self.context_config.heading_chains else []), *semantic]
        # Structural units have priority. Unfit semantic intervals must not
        # coalesce the entire list and thereby destroy shorter feasible links.
        intervals: list[tuple[int, int]] = []
        for link in self.links:
            size = self._count(document.canonical_text[link["start"]:link["end"]])
            link["tokens"] = size
            link["fits_target"] = size <= self.packing_config.target_tokens
            link["fits_encoder"] = size <= self.config.max_tokens
            if link["kind"] == "intro" and not link["fits_encoder"]:
                link["packing_decision"] = "token_limit_keep_legal_base_units"
                continue
            left = positions[link["a_block_ids"][0]]
            right = positions[link["b_block_id"]]
            overlapping = [(a, b) for a, b in intervals if a <= right and b >= left]
            union_left = min([left, *[a for a, _ in overlapping]])
            union_right = max([right, *[b for _, b in overlapping]])
            if link["kind"] == "intro" and self._count(document.canonical_text[
                    blocks[union_left].char_start:blocks[union_right].char_end]) > self.config.max_tokens:
                link["packing_decision"] = "overlapping_preferred_units_exceed_encoder_structural_priority"
                continue
            intervals = [(a, b) for a, b in intervals if (a, b) not in overlapping]
            intervals.append((union_left, union_right))
            link["packing_decision"] = "continuous_unit" if link["fits_encoder"] else "inherited_token_fallback"
        intervals.sort()
        end_by_start = dict(intervals)
        units, index = [], 0
        while index < len(blocks):
            end = end_by_start.get(index, index)
            units.append(Unit(blocks[index:end + 1], keys[index], keys[end]))
            index = end + 1
        return units

    def chunk(self, document: Document):
        if not self.context_config.heading_chains and not self.context_config.intro_rule:
            self.links, self.candidates = [], []
            return super().chunk(document)
        validate_chunking_document(document)
        self.diagnostics = []
        canonical = document.canonical_text
        result, active = [], []
        active_key = None
        for unit in self._context_units(document):
            start, end = unit.blocks[0].char_start, unit.blocks[-1].char_end
            size = self._count(canonical[start:end])
            if active and active_key != unit.first_key:
                result.append(self._make_chunk(document, active[0].char_start, active[-1].char_end, fallback=False))
                active = []
            if size > self.config.max_tokens:
                if active:
                    result.append(self._make_chunk(document, active[0].char_start, active[-1].char_end, fallback=False))
                    active = []
                # A long ancestor must not destroy a shorter feasible descendant
                # heading/body link. Keep the earliest fitting heading suffix;
                # split only its prefix using the unchanged token fallback.
                suffix = next((b for b in unit.blocks[1:] if b.type == "heading"
                               and self._count(canonical[b.char_start:end]) <= self.config.max_tokens), None)
                prefix_end = suffix.char_start if suffix else end
                view = Block(block_id=unit.blocks[0].block_id, type="paragraph", text=canonical[start:prefix_end],
                             char_start=start, char_end=prefix_end, section_path=unit.blocks[0].section_path)
                result.extend(self._split_block(document, view))
                if suffix:
                    result.append(self._make_chunk(document, suffix.char_start, end, fallback=False))
                self.diagnostics.append({"kind": "heading", "block_id": unit.blocks[0].block_id,
                                         "reason": "preferred_unit_exceeds_encoder_inherited_token_fallback",
                                         "start": start, "end": end, "tokens": size,
                                         "preserved_descendant_suffix_start": suffix.char_start if suffix else None})
                active_key = None
                continue
            if active and self._count(canonical[active[0].char_start:end]) > self.packing_config.target_tokens:
                result.append(self._make_chunk(document, active[0].char_start, active[-1].char_end, fallback=False))
                overlap = []
                for previous in reversed(active):
                    trial = [previous, *overlap]
                    if self._count(canonical[trial[0].char_start:trial[-1].char_end]) > self.config.overlap_tokens:
                        break
                    overlap = trial
                limit = max(self.packing_config.target_tokens, size)
                while overlap and self._count(canonical[overlap[0].char_start:end]) > limit:
                    overlap.pop(0)
                active = overlap
            active.extend(unit.blocks)
            active_key = unit.last_key
        if active:
            result.append(self._make_chunk(document, active[0].char_start, active[-1].char_end, fallback=False))
        for link in self.links:
            link["preserved_in_index"] = any(c.char_start <= link["start"] and c.char_end >= link["end"] for c in result)
            if not link["preserved_in_index"]:
                self.diagnostics.append({"kind": link["kind"], "block_id": link["b_block_id"],
                                         "link_id": link["link_id"], "reason": link["packing_decision"],
                                         "tokens": link["tokens"]})
        return result
