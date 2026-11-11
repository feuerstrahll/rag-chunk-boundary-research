"""Greedy canonical-span packing with optional immediate-parent boundaries.

Imports deliberately target the frozen upstream snapshot selected by the runner.
No parser blocks or canonical text are modified.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from chunking.fixed import TokenizerConfig, validate_chunking_document
from chunking.structure import StructureAwareChunker, StructureChunkingConfig
from models import Block, Chunk, Document


@dataclass(frozen=True)
class HierarchyPackingConfig:
    tokenizer: TokenizerConfig
    target_tokens: int = 480
    max_tokens: int = 510
    overlap_tokens: int = 64
    respect_parent_boundary: bool = True

    def __post_init__(self) -> None:
        if not 0 <= self.overlap_tokens < self.target_tokens <= self.max_tokens:
            raise ValueError("Require 0 <= overlap < target <= encoder text budget")


class HierarchyAwarePackingChunker(StructureAwareChunker):
    """Pack adjacent blocks; section identity comes from heading position.

    The inherited chunk constructor and oversized token fallback are unchanged.
    The ablation switches only the region partition, including root boundaries.
    """

    def __init__(self, config: HierarchyPackingConfig, *, tokenizer: Any) -> None:
        super().__init__(
            StructureChunkingConfig(
                tokenizer=config.tokenizer, max_tokens=config.max_tokens,
                overlap_tokens=config.overlap_tokens,
            ), tokenizer=tokenizer,
        )
        self.packing_config = config
        identity = {
            **super().provenance, "strategy": "hierarchy_packing",
            "algorithm_version": "1", "target_tokens": config.target_tokens,
            "respect_parent_boundary": config.respect_parent_boundary,
            "root_siblings_allowed": True,
            "heading_binding": "whole_unit_if_fits_hard_limit_else_inherited_fallback",
        }
        identity.pop("chunker_config_hash")
        self.chunker_config_hash = sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        self.diagnostics: list[dict[str, Any]] = []

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            **super().provenance, "strategy": "hierarchy_packing",
            "target_tokens": self.packing_config.target_tokens,
            "respect_parent_boundary": self.packing_config.respect_parent_boundary,
            "root_siblings_allowed": True, "algorithm_version": "1",
        }

    def _regions(self, document: Document) -> list[list[Block]]:
        stack: list[tuple[int, int]] = []
        regions: list[list[Block]] = []
        previous: tuple[int, int] | None = None
        for index, block in enumerate(document.blocks):
            if not block.text:
                continue
            if block.type == "heading":
                if block.level is None:
                    raise ValueError("Heading level is required")
                while stack and stack[-1][1] >= block.level:
                    stack.pop()
                stack.append((index, block.level))
            key = (stack[0][0], stack[-2][0] if len(stack) > 1 else -1) if stack else (-1, -1)
            if not self.packing_config.respect_parent_boundary:
                key = (-1, -1)
            if not regions or key != previous:
                regions.append([])
            regions[-1].append(block)
            previous = key
        return regions

    def _units(self, region: list[Block], *, last_region: bool) -> list[list[Block]]:
        """Bind a heading run to the following body before greedy packing."""
        units: list[list[Block]] = []
        pending: list[Block] = []
        for block in region:
            if block.type == "heading":
                pending.append(block)
            else:
                units.append([*pending, block])
                pending = []
        if pending:
            units.append(pending)
            self.diagnostics.append({
                "block_id": pending[0].block_id,
                "reason": "no_following_content" if last_region else "structural_boundary",
            })
        return units

    def chunk(self, document: Document) -> list[Chunk]:
        validate_chunking_document(document)
        self.diagnostics = []
        canonical = document.canonical_text
        result: list[Chunk] = []
        regions = self._regions(document)
        for region_index, region in enumerate(regions):
            active: list[Block] = []
            for unit in self._units(region, last_region=region_index == len(regions) - 1):
                start, end = unit[0].char_start, unit[-1].char_end
                unit_text = canonical[start:end]
                size = self._count(unit_text)
                if size > self.config.max_tokens:
                    if active:
                        result.append(self._make_chunk(document, active[0].char_start, active[-1].char_end, fallback=False))
                        active = []
                    if unit[0].type == "heading":
                        heading_end = next((b.char_start for b in unit if b.type != "heading"), end)
                        self.diagnostics.append({
                            "block_id": unit[0].block_id,
                            "reason": "heading_exceeds_limit" if self._count(canonical[start:heading_end]) >= self.config.max_tokens
                            else "heading_body_oversized_fallback",
                        })
                    # Synthetic fallback input is a view of the original span;
                    # resulting chunks recover original block IDs and anchors.
                    view = Block(
                        block_id=unit[0].block_id, type="paragraph", text=unit_text,
                        char_start=start, char_end=end, section_path=unit[0].section_path,
                    )
                    result.extend(self._split_block(document, view))
                    continue
                limit = max(self.packing_config.target_tokens, size)
                if not active:
                    active = list(unit)
                    continue
                if self._count(canonical[active[0].char_start:end]) <= self.packing_config.target_tokens:
                    active.extend(unit)
                    continue
                result.append(self._make_chunk(document, active[0].char_start, active[-1].char_end, fallback=False))
                overlap: list[Block] = []
                for previous in reversed(active):
                    trial = [previous, *overlap]
                    if self._count(canonical[trial[0].char_start:trial[-1].char_end]) > self.config.overlap_tokens:
                        break
                    overlap = trial
                while overlap and self._count(canonical[overlap[0].char_start:end]) > limit:
                    overlap.pop(0)
                active = [*overlap, *unit]
            if active:
                result.append(self._make_chunk(document, active[0].char_start, active[-1].char_end, fallback=False))
        return result
