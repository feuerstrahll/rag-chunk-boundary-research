"""Freeze a small boundary-focused diagnostic sample before content review."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq


def main() -> None:
    cli = argparse.ArgumentParser()
    cli.add_argument("--documents", type=Path, required=True)
    cli.add_argument("--chunks", type=Path, required=True)
    cli.add_argument("--out", type=Path, required=True)
    args = cli.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    docs = {d["doc_id"]: d for d in map(json.loads, args.documents.read_text(encoding="utf-8").splitlines())}
    chunks = defaultdict(list)
    for c in pq.read_table(args.chunks, columns=["doc_id", "char_start", "char_end"]).to_pylist():
        if c["doc_id"] in docs:
            chunks[c["doc_id"]].append(c)
    pools = {"between_blocks": [], "inside_oversized_block": []}
    for doc_id, doc in sorted(docs.items()):
        blocks = doc["blocks"]
        ends = {c["char_end"] for c in chunks[doc_id]}
        for offset in sorted(ends):
            if offset >= len(doc["canonical_text"]):
                continue
            inside = next((b for b in blocks if b["char_start"] < offset < b["char_end"]), None)
            if inside:
                category = "inside_oversized_block"
                nearby = [inside]
            else:
                category = "between_blocks"
                before = [b for b in blocks if b["char_end"] <= offset]
                after = [b for b in blocks if b["char_start"] >= offset]
                if not before or not after:
                    continue
                nearby = [before[-1], after[0]]
            pools[category].append({"doc_id": doc_id, "offset": offset,
                                    "block_ids": [b["block_id"] for b in nearby]})
    rng = random.Random(20261022)
    selected = []
    for category, quota in (("between_blocks", 8), ("inside_oversized_block", 4)):
        pool = pools[category]
        selected.extend({"category": category, **item} for item in rng.sample(pool, min(quota, len(pool))))
    output = {"status": "selection_frozen_before_content_review", "purpose": "diagnostic only; no prevalence estimate",
              "seed": 20261022, "rule": "uniform seeded sample of 8 actual between-block ends and 4 ends inside blocks, from pilot documents; no labels used",
              "pool_sizes": {key: len(value) for key, value in pools.items()},
              "inputs_sha256": {"documents": hashlib.sha256(args.documents.read_bytes()).hexdigest(),
                                "chunks": hashlib.sha256(args.chunks.read_bytes()).hexdigest()},
              "selected": selected}
    args.out.parent.mkdir(parents=True, exist_ok=False)
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output["pool_sizes"]))


if __name__ == "__main__":
    main()
