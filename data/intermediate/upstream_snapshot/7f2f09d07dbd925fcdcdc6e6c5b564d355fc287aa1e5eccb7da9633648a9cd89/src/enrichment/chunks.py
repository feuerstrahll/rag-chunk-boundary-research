"""Attach verifiable source metadata without changing chunk text or offsets."""

from models import Chunk, ChunkMetadata, Document
from ragstudy.data.interfaces import DocumentSource


def enrich_chunks(chunks: list[Chunk], document: Document, source: DocumentSource) -> list[Chunk]:
    blocks = {block.block_id: block for block in document.blocks}
    enriched: list[Chunk] = []
    for chunk in chunks:
        anchors = tuple(
            blocks[block_id].source_anchor
            for block_id in chunk.block_ids
            if blocks[block_id].source_anchor is not None
        )
        enriched.append(
            chunk.model_copy(
                update={
                    "metadata": ChunkMetadata(
                        title=source.metadata.get("title"),
                        source_url=source.provenance.get("source_url") or document.source,
                        representation=source.provenance.get("representation"),
                        source_format=source.source_format,
                        source_anchors=anchors,
                    )
                }
            )
        )
    return enriched
