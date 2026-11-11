"""Replay saved rankings and check the completed run without rerunning retrieval."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pyarrow.parquet as pq

ARTICLE = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ARTICLE / "src"))


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate(folder):
    from article_research.packing_evaluation import (
        METRICS,
        evidence_metrics,
        select_ranked_context,
    )

    manifest = read(folder / "manifest.json")
    if manifest["status"] != "completed_exploratory_exposed_ai_labels":
        raise ValueError("Not a completed full exploratory run")
    for relative, expected in manifest["artifact_sha256"].items():
        if sha(folder / relative) != expected:
            raise ValueError(f"Artifact hash mismatch: {relative}")
    config, preflight, summary = [read(folder / name) for name in ["frozen_config.json", "preflight.json", "summary.json"]]
    if sha(folder / "frozen_config.json") != preflight["config_sha256"]:
        raise ValueError("Frozen config hash mismatch")
    gold_file = (ARTICLE / config["gold"]["path"]).resolve()
    gold = {r["question_id"]: r for r in map(json.loads, gold_file.read_text().splitlines())}
    if sha(gold_file) != config["gold"]["sha256"] or len(gold) != 118:
        raise ValueError("Frozen gold population mismatch")
    metrics = pq.read_table(folder / "per_question_metrics.parquet").to_pylist()
    if len(metrics) != 1416:
        raise ValueError("Incorrect detailed-table size")
    lookup = {(r["variant"], r["method"], r["query_id"]): r for r in metrics}
    expected = {(variant, method, qid) for variant in config["variants"] for method in ["bm25", "dense", "hybrid"] for qid in gold}
    if set(lookup) != expected:
        raise ValueError("Paired matrix mismatch")
    checked_rankings, checked_chunks, maximum_context = 0, 0, 0
    for variant in config["variants"]:
        path = folder / variant
        statistics = read(path / "index_statistics.json")
        chunks = [SimpleNamespace(**row) for row in pq.read_table(path / "chunks.parquet").to_pylist()]
        checked_chunks += len(chunks)
        if len(chunks) != statistics["chunk_count"] or sum(c.token_count for c in chunks) != statistics["indexed_tokens"]:
            raise ValueError("Chunk count/token cost mismatch")
        hard = 480 if variant == "fixed" else statistics["encoder_text_budget"]
        if any(c.token_count > hard for c in chunks):
            raise ValueError("Saved encoder text-budget violation")
        if math.prod(statistics["dense_matrix_shape"]) * 4 != statistics["dense_matrix_bytes"]:
            raise ValueError("Dense float32 matrix size mismatch")
        if statistics["uncovered_nonwhitespace_characters"]:
            raise ValueError("Uncovered nonwhitespace text")
        contexts = {(r["query_id"], r["method"]): r for r in map(json.loads, (path / "contexts.jsonl").read_text().splitlines())}
        parquet = pq.ParquetFile(path / "rankings.parquet")
        for group in range(parquet.num_row_groups):
            rows = parquet.read_row_group(group).to_pylist()
            if not rows:
                raise ValueError("Empty ranking row group")
            qid, method = rows[0]["query_id"], rows[0]["method"]
            if any(r["query_id"] != qid or r["method"] != method for r in rows):
                raise ValueError("Unexpected ranking group layout")
            ranking = [SimpleNamespace(chunk=chunks[r["chunk_index"]], rank=r["rank"], score=r["score"]) for r in rows]
            if [r.rank for r in ranking] != list(range(1, len(ranking) + 1)):
                raise ValueError("Nonconsecutive ranking")
            keys = [(-r.score, r.chunk.doc_id, r.chunk.chunk_id) for r in ranking]
            if keys != sorted(keys):
                raise ValueError("Score/tie order mismatch")
            if method == "dense" and len(ranking) != len(chunks):
                raise ValueError("Dense ranking was cut before document projection")
            question = gold[qid]
            spans = [(s["char_start"], s["char_end"]) for s in question["gold_spans"]]
            scores = evidence_metrics(ranking, question["doc_id"], spans)
            saved = lookup[(variant, method, qid)]
            for name, value in scores.items():
                if not math.isclose(value, saved[name], rel_tol=0, abs_tol=1e-12):
                    raise ValueError(f"Evidence replay mismatch: {variant}/{method}/{qid}/{name}")
            # Independent equivalent max-pool projection: first occurrence in
            # the fully sorted ranking is the maximum-scoring chunk per doc.
            unique = list(dict.fromkeys(r.chunk.doc_id for r in ranking))
            position = unique.index(question["doc_id"]) + 1 if question["doc_id"] in unique else None
            if saved["DocumentRecall@5"] != float(position is not None and position <= 5):
                raise ValueError("DocumentRecall replay mismatch")
            if saved["DocumentMRR@10"] != (1 / position if position is not None and position <= 10 else 0):
                raise ValueError("DocumentMRR replay mismatch")
            selected, used = select_ranked_context(ranking, 1024)
            context = contexts[(qid, method)]
            if used > 1024 or used != context["used_tokens"]:
                raise ValueError("Returned context cost mismatch")
            if [r.chunk.chunk_id for r in selected] != [r["chunk_id"] for r in context["chunks"]]:
                raise ValueError("Returned context selection mismatch")
            maximum_context = max(maximum_context, used)
            checked_rankings += 1
    if checked_rankings != 1416 or summary["cluster_count"] != 118:
        raise ValueError("Ranking or cluster count mismatch")
    grouped = defaultdict(list)
    for row in metrics:
        grouped[(row["variant"], row["method"])].append(row)
    for row in summary["aggregates"]:
        for metric in METRICS:
            values = grouped[(row["variant"], row["method"])]
            if not math.isclose(row[metric], sum(r[metric] for r in values) / len(values), abs_tol=1e-12):
                raise ValueError("Aggregate mismatch")
    return {
        "status": "passed", "run": str(folder), "manifest_sha256": sha(folder / "manifest.json"),
        "checked_per_question_rows": len(metrics), "checked_full_rankings": checked_rankings,
        "checked_chunks": checked_chunks, "document_clusters": 118,
        "maximum_returned_context_tokens": maximum_context,
        "checks": ["artifact_hashes", "paired_population", "ranking_depth_and_ties", "metric_replay",
                   "document_max_pool_before_cutoff", "context_selection_and_repeated_token_cost",
                   "aggregate_recalculation", "chunk_cost_and_encoder_budget", "dense_matrix_bytes"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite validation")
    result = validate(args.run.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
