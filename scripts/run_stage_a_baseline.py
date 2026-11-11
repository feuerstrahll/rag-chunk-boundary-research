"""Run frozen structure or selected fixed Markdown baseline without network access."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any, Self

import orjson
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
from prepare_baseline_snapshot import sha256_file, validate_snapshot

METHODS = ("bm25", "dense", "hybrid")
PACKAGE_NAMES = (
    "datasets", "numpy", "orjson", "pyarrow", "pydantic",
    "sentence-transformers", "tokenizers", "torch", "transformers",
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def validate_run_dir(path: Path, article: Path) -> Path:
    resolved = path.resolve()
    allowed = (article / "results" / "runs").resolve()
    if resolved == allowed or allowed not in resolved.parents:
        raise ValueError(f"Run directory must be inside {allowed}")
    if resolved.exists():
        raise FileExistsError(f"Refusing to overwrite run directory: {resolved}")
    return resolved


def validate_mode(mode: str, allow_historical_regression: bool) -> None:
    if mode == "historical_regression" and not allow_historical_regression:
        raise PermissionError("historical_regression requires --allow-historical-regression")


def resolve_input_paths(raw_root: Path, dataset: dict[str, Any], mode: str) -> dict[str, Path]:
    paths = {
        "corpus": raw_root / dataset["corpus"],
        "train_qa": raw_root / dataset["train_qa"],
    }
    if mode == "historical_regression":
        paths["historical_test_qa"] = raw_root / dataset["historical_test_qa"]
    return paths


def local_model_snapshot(model: str, revision: str) -> Path:
    cache_name = "models--" + model.replace("/", "--")
    path = Path.home() / ".cache" / "huggingface" / "hub" / cache_name / "snapshots" / revision
    if not path.is_dir():
        raise FileNotFoundError(f"Pinned model snapshot is not local: {path}")
    return path.resolve()


def environment() -> dict[str, Any]:
    packages = {}
    for name in PACKAGE_NAMES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "packages": packages,
        "network_policy": "offline_local_files_only",
    }


class ResourceSampler:
    def __init__(self) -> None:
        self.process = psutil.Process()
        self.peak_rss = self.process.memory_info().rss
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self.stop_event.wait(0.01):
            self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)

    def __enter__(self) -> Self:
        self.thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop_event.set()
        self.thread.join()
        self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    table = pa.Table.from_pylist(rows)
    pq.write_table(
        table, path, compression="zstd", use_dictionary=False, write_statistics=True,
        version="2.6", data_page_version="1.0", row_group_size=max(1, len(rows)),
    )


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: fmean(row[key] for row in rows) for key in rows[0]} if rows else {}


def chunk_rows(chunks: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "text": chunk.text,
            "block_ids": list(chunk.block_ids),
            "pages": list(chunk.pages),
            "section_path": list(chunk.section_path),
            "block_types": list(chunk.block_types),
            "char_start": chunk.char_start,
            "char_end": chunk.char_end,
            "token_count": chunk.token_count,
            "oversize": chunk.oversize,
            "fallback_split": chunk.fallback_split,
            "metadata_json": canonical_json(chunk.metadata.model_dump(mode="json") if chunk.metadata else None),
        }
        for chunk in chunks
    ]


def historical_jsonl_hash(chunks: list[Any]) -> str:
    content = b"".join(
        orjson.dumps(chunk.model_dump(mode="json"), option=orjson.OPT_APPEND_NEWLINE) for chunk in chunks
    )
    return hashlib.sha256(content).hexdigest()


def compare_runs(current: Path, reference: Path | None) -> dict[str, Any]:
    if reference is None:
        return {"status": "pending_pair", "reference_run": None, "checks": {}}
    required = ("chunks.parquet", "retrieval_runs.parquet", "index_statistics.json")
    checks = {}
    for name in required:
        current_path, reference_path = current / name, reference / name
        if not reference_path.is_file():
            raise FileNotFoundError(f"Reference run artifact missing: {reference_path}")
        checks[name] = {
            "current_sha256": sha256_file(current_path),
            "reference_sha256": sha256_file(reference_path),
            "identical": sha256_file(current_path) == sha256_file(reference_path),
        }
    return {
        "status": "identical" if all(item["identical"] for item in checks.values()) else "different",
        "reference_run": str(reference.resolve()),
        "checks": checks,
    }


def run(args: argparse.Namespace) -> Path:
    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"})
    sys.dont_write_bytecode = True
    article = Path(__file__).resolve().parents[1]
    run_dir = validate_run_dir(args.run_dir, article)
    snapshot = args.snapshot.resolve()
    snapshot_manifest = validate_snapshot(snapshot)
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_mode(args.mode, args.allow_historical_regression)
    started_at_utc = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    run_dir.mkdir(parents=True)
    (run_dir / "source_snapshot_manifest.json").write_text(
        json.dumps(snapshot_manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8", newline="\n"
    )

    sys.path.insert(0, str(snapshot / "src"))
    from chunking.fixed import FixedChunkingConfig, FixedTokenChunker, TokenizerConfig
    from chunking.structure import StructureAwareChunker, StructureChunkingConfig
    from enrichment.chunks import enrich_chunks
    from evaluation.retrieval import (
        align_evidence,
        document_scores,
        evidence_scores,
        integrity_statistics,
    )
    from parsers.markdown import MarkdownParser
    from ragstudy.data.watsonx_docsqa import WatsonxDocsQACorpusLoader
    from ragstudy.data.watsonx_qa import load_qa_parquets
    from retrieval.ranking import (
        BM25Index,
        DenseIndex,
        reciprocal_rank_fusion,
        resolve_text_budget,
    )
    from sentence_transformers import SentenceTransformer

    random.seed(config["seed"])
    upstream = Path(snapshot_manifest["upstream_path"])
    dataset = config["dataset"]
    raw_root = upstream / "data" / "raw" / "watsonx_docsqa" / dataset["revision"]
    paths = resolve_input_paths(raw_root, dataset, args.mode)
    corpus_path = paths["corpus"]
    train_path = paths["train_qa"]
    expected = {name: dataset[f"{name}_sha256"] for name in paths}
    qa_paths = {"train": train_path}
    if args.mode == "historical_regression":
        qa_paths["test"] = paths["historical_test_qa"]
    input_hashes = {name: sha256_file(path) for name, path in paths.items()}
    if input_hashes != expected:
        raise ValueError(f"Input hash mismatch: expected {expected}, got {input_hashes}")

    stages: dict[str, float] = {}
    started = time.perf_counter()
    with ResourceSampler() as resources:
        stage = time.perf_counter()
        loader = WatsonxDocsQACorpusLoader.from_parquet(corpus_path, revision=dataset["revision"])
        if loader.row_count != dataset["documents"]:
            raise ValueError("Corpus row count differs from frozen config")
        corpus_ids = set(loader.dataset["doc_id"])
        questions = load_qa_parquets(qa_paths, corpus_doc_ids=corpus_ids)
        if args.mode == "stage_a_train" and any(question.split != "train" for question in questions):
            raise ValueError("stage_a_train loaded a non-train question")
        if args.smoke:
            questions = questions[:2]
            smoke_ids = set().union(*(question.gold_doc_ids for question in questions))
            if len(smoke_ids) != 2:
                raise ValueError(f"Smoke selection must resolve to exactly 2 documents, got {len(smoke_ids)}")
        else:
            smoke_ids = None
        stages["load_inputs_seconds"] = time.perf_counter() - stage

        stage = time.perf_counter()
        sources = loader.documents("md_document")
        if smoke_ids is not None:
            sources = [source for source in sources if source.document_id in smoke_ids]
        parser = MarkdownParser()
        documents = [
            parser.parse_text(
                str(source.content), document_id=source.document_id,
                source_label=(
                    f"hf://datasets/{dataset['name']}@{dataset['revision']}/"
                    f"{source.document_id}/md_document"
                ),
                source_url=source.provenance.get("source_url"),
            )
            for source in sources
        ]
        documents_by_id = {document.doc_id: document for document in documents}
        stages["parse_seconds"] = time.perf_counter() - stage

        model_snapshot = local_model_snapshot(config["embedding"]["model"], config["embedding"]["revision"])
        stage = time.perf_counter()
        model = SentenceTransformer(str(model_snapshot), device="cpu", local_files_only=True)
        text_budget = resolve_text_budget(
            model, requested_max_tokens=config["embedding"]["requested_max_tokens"]
        )
        special_tokens = model.tokenizer.num_special_tokens_to_add(pair=False)
        encoder_limit = min(int(model.max_seq_length), int(model.tokenizer.model_max_length))
        if text_budget != encoder_limit - special_tokens:
            raise ValueError("Resolved chunk budget does not equal encoder limit minus special tokens")
        stages["model_load_seconds"] = time.perf_counter() - stage

        tokenizer_config = TokenizerConfig(
            name=config["embedding"]["model"], revision=config["embedding"]["revision"]
        )
        if args.segmenter == "structure":
            chunker = StructureAwareChunker(
                StructureChunkingConfig(
                    strategy="structure", tokenizer=tokenizer_config, max_tokens=text_budget,
                    overlap_tokens=config["chunking"]["overlap_tokens"],
                ), tokenizer=model.tokenizer,
            )
        else:
            fixed = config["comparison_fixed"]
            if fixed["candidate_id"] != "fixed_m480_o20" or fixed["max_tokens"] > text_budget:
                raise ValueError("Selected fixed configuration is not pinned or exceeds encoder budget")
            chunker = FixedTokenChunker(
                FixedChunkingConfig(
                    strategy="fixed", tokenizer=tokenizer_config,
                    max_tokens=fixed["max_tokens"], overlap_tokens=fixed["overlap_tokens"],
                ), tokenizer=model.tokenizer,
            )
        stage = time.perf_counter()
        chunks = [
            chunk
            for source, document in zip(sources, documents)
            for chunk in enrich_chunks(chunker.chunk(document), document, source)
        ]
        for chunk in chunks:
            document = documents_by_id[chunk.doc_id]
            if chunk.text != document.canonical_text[chunk.char_start : chunk.char_end]:
                raise ValueError(f"Chunk round-trip failed: {chunk.chunk_id}")
        stages["chunk_seconds"] = time.perf_counter() - stage

        stage = time.perf_counter()
        bm25 = BM25Index(chunks)
        dense = DenseIndex(chunks, model, batch_size=config["embedding"]["batch_size"])
        stages["index_seconds"] = time.perf_counter() - stage

        aligned = {
            question.query_id: [align_evidence(item, documents_by_id[item.doc_id]) for item in question.evidence]
            for question in questions
        }
        if any(not item.spans for items in aligned.values() for item in items):
            raise ValueError("Baseline contains unaligned evidence")
        by_doc: dict[str, list[Any]] = defaultdict(list)
        for chunk in chunks:
            by_doc[chunk.doc_id].append(chunk)
        retrieval_rows: list[dict[str, Any]] = []
        metric_rows: dict[str, dict[str, list[dict[str, float]]]] = {
            method: {"document": [], "evidence": []} for method in METHODS
        }
        stage = time.perf_counter()
        for question in questions:
            bm25_ranking = bm25.search(question.question)
            dense_ranking = dense.search(question.question)
            hybrid_ranking = reciprocal_rank_fusion(
                bm25_ranking, dense_ranking,
                constant=config["retrieval"]["fusion_constant"], depth=config["retrieval"]["fusion_depth"],
            )
            for method, ranking in zip(METHODS, (bm25_ranking, dense_ranking, hybrid_ranking)):
                document_metric = document_scores(ranking, question.gold_doc_ids, top_k=config["retrieval"]["top_k"])
                evidence_items = [
                    evidence_scores(
                        item, ranking, by_doc[item.evidence.doc_id], top_k=config["retrieval"]["top_k"],
                        hit_threshold=config["evaluation"]["evidence_hit_threshold"],
                    )
                    for item in aligned[question.query_id]
                ]
                evidence_metric = mean_metrics(evidence_items)
                metric_rows[method]["document"].append(document_metric)
                metric_rows[method]["evidence"].append(evidence_metric)
                retrieval_rows.append(
                    {
                        "query_id": question.query_id,
                        "split": question.split,
                        "method": method,
                        "top_chunks_json": canonical_json([
                            {"rank": item.rank, "chunk_id": item.chunk.chunk_id,
                             "doc_id": item.chunk.doc_id, "score": item.score}
                            for item in ranking[: config["retrieval"]["ranking_depth"]]
                        ]),
                        "document_metrics_json": canonical_json(document_metric),
                        "evidence_metrics_json": canonical_json(evidence_metric),
                    }
                )
        stages["query_seconds"] = time.perf_counter() - stage

    chunks_path = run_dir / "chunks.parquet"
    retrieval_path = run_dir / "retrieval_runs.parquet"
    write_parquet(chunks_path, chunk_rows(chunks))
    write_parquet(retrieval_path, retrieval_rows)
    metrics = {
        method: {level: mean_metrics(rows) for level, rows in levels.items()}
        for method, levels in metric_rows.items()
    }
    integrity = integrity_statistics(documents, chunks)
    index_statistics = {
        "documents": len(documents),
        "questions": len(questions),
        "chunks": len(chunks),
        "total_chunk_tokens": sum(chunk.token_count for chunk in chunks),
        "semantic_chunks_jsonl_sha256": historical_jsonl_hash(chunks),
        "integrity": integrity,
        "metrics": metrics,
    }
    (run_dir / "index_statistics.json").write_text(
        json.dumps(index_statistics, indent=2, ensure_ascii=True) + "\n", encoding="utf-8", newline="\n"
    )
    execution = {
        "stages_seconds": stages,
        "total_seconds": time.perf_counter() - started,
        "peak_rss_bytes": resources.peak_rss,
        "rss_sampling_interval_seconds": 0.01,
        "device": "cpu",
    }
    (run_dir / "execution_cost.json").write_text(
        json.dumps(execution, indent=2, ensure_ascii=True) + "\n", encoding="utf-8", newline="\n"
    )
    reproducibility = compare_runs(run_dir, args.compare_to.resolve() if args.compare_to else None)
    (run_dir / "reproducibility_check.json").write_text(
        json.dumps(reproducibility, indent=2, ensure_ascii=True) + "\n", encoding="utf-8", newline="\n"
    )

    historical_summary_path = upstream / "results" / "watsonx_retrieval" / "summary.json"
    historical_structure_path = upstream / "results" / "watsonx_retrieval" / "md_document" / "structure" / "chunks.jsonl"
    historical_summary = json.loads(historical_summary_path.read_text(encoding="utf-8"))
    historical_train = historical_summary["representations"]["md_document"]["structure"]["retrieval"]["hybrid"]["evidence"]["by_split"]["train"]
    current_coverage = metrics["hybrid"]["evidence"]["EvidenceCoverage@5"]
    chunks_match = (index_statistics["semantic_chunks_jsonl_sha256"] == sha256_file(historical_structure_path)
                    if args.segmenter == "structure" else None)
    comparison = [
        "# Historical comparison", "",
        f"Mode: `{args.mode}`; smoke: `{str(args.smoke).lower()}`.", "",
        "Reproducibility classification: `current_implementation_baseline`.", "",
        "Exact reproduction is not claimed: historical manifests came from different dirty revisions and do not preserve their patches.", "",
        f"Current chunks: {len(chunks)}; tokens: {index_statistics['total_chunk_tokens']}.",
        "Historical structure chunks: 6007; tokens: 1316605.",
        f"Segmenter: `{args.segmenter}`. Structure historical chunk-hash match: `{str(chunks_match).lower()}`.",
        f"Current train Hybrid EvidenceCoverage@5 ({args.segmenter}): {current_coverage:.12f}.",
        f"Stored historical structure train Hybrid EvidenceCoverage@5: {historical_train['EvidenceCoverage@5']:.12f}.",
        "Historical metric is context only; fixed-versus-structure inference requires paired current runs.",
        "",
        "The historical 30-row test split was not loaded in stage_a_train mode. Historical regression remains an explicit opt-in technical mode.",
    ]
    (run_dir / "historical_comparison.md").write_text("\n".join(comparison) + "\n", encoding="utf-8", newline="\n")

    artifact_names = (
        "source_snapshot_manifest.json", "chunks.parquet", "retrieval_runs.parquet",
        "index_statistics.json", "execution_cost.json", "reproducibility_check.json",
        "historical_comparison.md",
    )
    manifest = {
        "schema_version": "1.0",
        "started_at_utc": started_at_utc,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "mode": args.mode,
        "segmenter": args.segmenter,
        "segmenter_config": (config["chunking"] if args.segmenter == "structure" else config["comparison_fixed"]),
        "smoke": args.smoke,
        "reproducibility_classification": "current_implementation_baseline",
        "classification_reason": (
            "Current frozen implementation ran successfully; exact historical source patches are unavailable."
        ),
        **config["contamination"],
        "previously_exposed": args.mode == "historical_regression",
        "source_snapshot_id": snapshot_manifest["snapshot_id"],
        "source_snapshot_git": snapshot_manifest["git"],
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "input_sha256": input_hashes,
        "environment": environment(),
        "model": {
            **config["embedding"], "local_snapshot": str(model_snapshot),
            "encoder_limit": encoder_limit, "special_tokens": special_tokens,
        },
        "budgets": {
            "max_chunk_length_tokens": (text_budget if args.segmenter == "structure"
                                        else config["comparison_fixed"]["max_tokens"]),
            "embedding_input_limit_tokens_including_special": encoder_limit,
            "embedding_special_tokens": special_tokens,
            "cumulative_retrieval_context_budget": config["retrieval"]["cumulative_context_budget"],
        },
        "retrieval": config["retrieval"],
        "artifacts": {name: sha256_file(run_dir / name) for name in artifact_names},
        "command": [str(Path(sys.executable).resolve()), *sys.argv],
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8", newline="\n"
    )
    return run_dir


def main() -> None:
    article = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=article / "configs" / "baseline.json")
    parser.add_argument("--mode", choices=("stage_a_train", "historical_regression"), default="stage_a_train")
    parser.add_argument("--allow-historical-regression", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--segmenter", choices=("structure", "fixed"), default="structure")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--compare-to", type=Path)
    args = parser.parse_args()
    print(run(args))


if __name__ == "__main__":
    main()
