"""Build a small blind block-pair audit queue from pinned train documents."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT.parent / "techdoc-chunking-pipeline"
REVISION = "f7da4b6d8a8c4609c49fda06af3bea29b6ef9be4"
DATA = UPSTREAM / "data" / "raw" / "watsonx_docsqa" / REVISION
CORPUS = DATA / "corpus" / "train-00000-of-00001.parquet"
QA = DATA / "question_answers" / "train-00000-of-00001.parquet"
EXPECTED = {
    "corpus": "fb56d7c847f576d72f32669b01feb47e38788b7e91deeecdcdbcc2b9995fdd6b",
    "qa": "73c81739e29b2e5fac82accb43d7b3a7f319e047ae56f421dd8d3804f0742bb1",
}
SEED = 20261022


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def detector(blocks: list[dict], index: int) -> list[str]:
    a, b = blocks[index - 1], blocks[index]
    text = b["text"].lower()
    tags = []
    if b["type"] == "list":
        tags.append("list")
    if b["type"] == "heading" or a["type"] == "heading":
        tags.append("heading_transition")
    if "`" in b["text"] or b["type"] == "code":
        tags.append("code_marker")
    if "|" in b["text"] or b["type"] == "table":
        tags.append("table_marker")
    if any(word in text for word in ("this ", "these ", "it ", "they ", "above", "following")):
        tags.append("reference_marker")
    return tags


def main() -> int:
    arg = argparse.ArgumentParser(description=__doc__)
    arg.add_argument("--output-dir", type=Path, required=True)
    args = arg.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    if (ROOT / "results" / "pilots").resolve() not in output.parents:
        raise ValueError("output-dir must be a new child of article/results/pilots")
    for name, path in (("corpus", CORPUS), ("qa", QA)):
        if digest(path) != EXPECTED[name]:
            raise ValueError(f"Pinned {name} hash mismatch")

    sys.path.insert(0, str(UPSTREAM / "src"))
    from parsers.markdown import MarkdownParser

    qa = pq.read_table(QA).to_pylist()
    corpus = pq.read_table(CORPUS, columns=["doc_id", "title", "md_document"]).to_pylist()
    by_id = {str(row["doc_id"]): row for row in corpus}
    def source_ids(row: dict) -> list[str]:
        raw = row.get("ground_truths_contexts_ids", row["correct_answer_document_ids"])
        values = json.loads(raw) if isinstance(raw, str) and raw.lstrip().startswith("[") else raw
        return [values] if isinstance(values, str) else list(values)

    development_ids = sorted({doc_id for row in qa for doc_id in source_ids(row)})
    parser = MarkdownParser()
    parsed = {}
    for doc_id in development_ids:
        row = by_id.get(doc_id)
        if row is None or not row["md_document"]:
            continue
        document = parser.parse_text(row["md_document"], document_id=doc_id, source_label=f"watsonxDocsQA:{doc_id}")
        blocks = [{"block_id": b.block_id, "type": b.type, "text": b.text,
                       "char_start": b.char_start, "char_end": b.char_end,
                       "section_path": list(b.section_path)} for b in document.blocks]
        if len(blocks) >= 7:
            parsed[doc_id] = {"doc_id": doc_id, "title": row["title"], "canonical_text": document.canonical_text,
                                  "source_markdown": row["md_document"], "blocks": blocks}
    rng = random.Random(SEED)
    eligible = sorted(parsed)
    if len(eligible) < 9:
        raise ValueError(f"Only {len(eligible)} eligible development documents")
    random_docs = rng.sample(eligible, 5)
    remaining = [doc_id for doc_id in eligible if doc_id not in random_docs]
    diagnostic_docs = []
    target_tags = ["list", "code_marker", "table_marker", "reference_marker"]
    for tag in target_tags:
        candidates = [doc_id for doc_id in remaining if doc_id not in diagnostic_docs and
                      any(tag in detector(parsed[doc_id]["blocks"], i)
                          for i in range(1, len(parsed[doc_id]["blocks"])))]
        if candidates:
            diagnostic_docs.append(candidates[0])
    for doc_id in remaining:
        if len(diagnostic_docs) == 4:
            break
        if doc_id not in diagnostic_docs:
            diagnostic_docs.append(doc_id)
    selected = random_docs + diagnostic_docs
    queue, predictions = [], []
    for doc_id in selected:
        doc = parsed[doc_id]
        blocks = doc["blocks"]
        all_indices = list(range(1, len(blocks)))
        if doc_id in random_docs:
            indices = sorted(rng.sample(all_indices, 6))
            stratum = "random"
        else:
            ranked = sorted(all_indices, key=lambda i: (-len(detector(blocks, i)), i))
            indices = sorted(ranked[:6])
            stratum = "diagnostic"
        for i in indices:
            a, b = blocks[i - 1], blocks[i]
            record_id = f"pilot_{len(queue) + 1:03d}"
            queue.append({"record_id": record_id, "doc_id": doc_id, "stratum": stratum,
                              "b_block_id": b["block_id"], "b_span": [b["char_start"], b["char_end"]],
                              "b_quote": b["text"], "candidate_a_block_id": a["block_id"],
                              "candidate_a_span": [a["char_start"], a["char_end"]]})
            predictions.append({"record_id": record_id, "selection_tags": detector(blocks, i)})
    docs = [parsed[doc_id] for doc_id in selected]
    for doc in docs:
        for block in doc["blocks"]:
            assert doc["canonical_text"][block["char_start"]:block["char_end"]] == block["text"]
    output.mkdir(parents=True)
    write_jsonl(output / "documents.jsonl", docs)
    write_jsonl(output / "queue.jsonl", queue)
    write_jsonl(output / "detector_predictions.jsonl", predictions)
    manifest = {"schema_version": "1.0", "status": "unlabeled_pilot", "seed": SEED,
                    "corpus_revision": REVISION, "input_sha256": EXPECTED, "train_qa_count": len(qa),
                    "development_document_count": len(development_ids), "eligible_document_count": len(eligible),
                    "random_document_ids": random_docs, "diagnostic_document_ids": diagnostic_docs,
                    "random_rule": "sample 5 eligible train-referenced documents by sorted doc_id and seed; sample 6 adjacent block boundaries uniformly per document without text markers or retrieval outcomes",
                    "diagnostic_rule": "choose 4 remaining train-referenced documents by sorted doc_id with list, code, table, reference markers where available; select 6 boundaries ranked by marker count then block index",
                    "queue_count": len(queue), "parser": "upstream src/parsers/markdown.py",
                    "builder_sha256": digest(Path(__file__)),
                    "parser_sha256": digest(UPSTREAM / "src" / "parsers" / "markdown.py"),
                    "parser_caveat": "Line-oriented parser normalizes list markers and joins paragraph/code lines; original Markdown is retained in documents.jsonl for verification",
                    "output_sha256": {name: digest(output / name) for name in
                                   ("documents.jsonl", "queue.jsonl", "detector_predictions.jsonl")}}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "documents": len(docs), "queue": len(queue)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
