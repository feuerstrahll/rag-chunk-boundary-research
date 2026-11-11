"""One frozen, offline, four-variant exploratory segmentation experiment.

Each variant runs in a new subprocess. Outputs are never resumed or overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

os.environ.update({"HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.dont_write_bytecode = True
ARTICLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ARTICLE / "src"))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def now():
    return datetime.now(timezone.utc).isoformat()


def resolve_inputs(config):
    snapshot = (ARTICLE / config["snapshot"]).resolve()
    corpus = (ARTICLE / config["dataset"]["corpus"]).resolve()
    gold = (ARTICLE / config["gold"]["path"]).resolve()
    model = Path(config["model"]["local_snapshot"]).resolve()
    for path in (snapshot, corpus, gold, model):
        if not path.exists():
            raise FileNotFoundError(f"Required pinned local artifact missing: {path}")
    from prepare_baseline_snapshot import validate_snapshot
    validate_snapshot(snapshot)
    for path, expected in [(corpus, config["dataset"]["corpus_sha256"]), (gold, config["gold"]["sha256"])]:
        if digest(path) != expected:
            raise ValueError(f"Input hash mismatch: {path}")
    if model.name != config["model"]["revision"] or not (model / "model.safetensors").exists():
        raise ValueError("Unexpected model revision or missing local weights")
    sys.path.insert(0, str(snapshot / "src"))
    return snapshot, corpus, gold, model


def load_documents(config, *, smoke=False):
    import pyarrow.parquet as pq
    from parsers.markdown import MarkdownParser
    from ragstudy.data.watsonx_docsqa import WatsonxDocsQACorpusLoader

    _, corpus_path, gold_path, _ = resolve_inputs(config)
    gold = [json.loads(line) for line in gold_path.read_text(encoding="utf-8").splitlines() if line]
    if len(gold) != config["gold"]["questions"] or len({q["question_id"] for q in gold}) != len(gold):
        raise ValueError("Frozen question population changed")
    if sum(len(row["gold_spans"]) for row in gold) != config["gold"]["spans"]:
        raise ValueError("Frozen gold-span count changed")
    if smoke:
        gold = sorted(gold, key=lambda q: q["question_id"])[:2]
    loader = WatsonxDocsQACorpusLoader.from_parquet(corpus_path, revision=config["dataset"]["revision"])
    if loader.row_count != config["dataset"]["documents"]:
        raise ValueError("Unexpected corpus population")
    sources = loader.documents("md_document")
    if smoke:
        selected = {q["doc_id"] for q in gold}
        sources = [source for source in sources if source.document_id in selected]
    parser = MarkdownParser()
    documents = [parser.parse_text(
        str(source.content), document_id=source.document_id,
        source_label="hierarchy_packing_frozen_markdown",
        source_url=source.provenance["source_url"],
    ) for source in sources]
    docs = {doc.doc_id: doc for doc in documents}
    for question in gold:
        text = docs[question["doc_id"]].canonical_text
        ordered = sorted(question["gold_spans"], key=lambda s: s["char_start"])
        for span in ordered:
            a, b = span["char_start"], span["char_end"]
            if not isinstance(a, int) or not isinstance(b, int) or not 0 <= a < b <= len(text):
                raise ValueError(f"Invalid gold offsets: {question['question_id']}")
            if text[a:b] != span["text"]:
                raise ValueError(f"Canonical gold mismatch: {question['question_id']}")
        if any(a["char_end"] > b["char_start"] for a, b in zip(ordered, ordered[1:])):
            raise ValueError("Gold spans overlap")
    raw_rows = pq.read_table(corpus_path).to_pylist() if not smoke else []
    return documents, {s.document_id: s for s in sources}, gold, raw_rows


def validate_chunks(document, chunks, tokenizer):
    """Require literal slices and coverage of every tokenizer-addressable char."""
    text = document.canonical_text
    covered = bytearray(len(text))
    ids = set()
    for chunk in chunks:
        if chunk.doc_id != document.doc_id or not 0 <= chunk.char_start < chunk.char_end <= len(text):
            raise ValueError("Invalid chunk identity/offsets")
        if chunk.chunk_id in ids or chunk.text != text[chunk.char_start:chunk.char_end]:
            raise ValueError("Duplicate chunk ID or canonical-slice mismatch")
        ids.add(chunk.chunk_id)
        covered[chunk.char_start:chunk.char_end] = b"\1" * (chunk.char_end - chunk.char_start)
        if chunk.token_count != len(tokenizer(chunk.text, add_special_tokens=False, truncation=False)["input_ids"]):
            raise ValueError("Incorrect token count")
    offsets = tokenizer(text, add_special_tokens=False, truncation=False, return_offsets_mapping=True)["offset_mapping"]
    if any(not all(covered[a:b]) for a, b in offsets if b > a):
        raise ValueError(f"Uncovered tokenizer-addressable text: {document.doc_id}")
    return {"uncovered_characters": sum(v == 0 for v in covered), "uncovered_nonwhitespace_characters":
            sum(not text[i].isspace() for i, v in enumerate(covered) if not v)}


class RssSampler:
    def __enter__(self):
        import psutil
        self.process = psutil.Process()
        self.peak = self.process.memory_info().rss
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.thread.start()
        return self

    def sample(self):
        while not self.stop.wait(0.1):
            self.peak = max(self.peak, self.process.memory_info().rss)

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()
        self.peak = max(self.peak, self.process.memory_info().rss)


def worker(args):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from article_research.packing_evaluation import (
        document_metrics,
        evidence_metrics,
        select_ranked_context,
    )

    output = args.output_dir.resolve()
    config = read_json(output / "frozen_config.json")
    preflight = read_json(output / "preflight.json")
    if digest(output / "frozen_config.json") != preflight["config_sha256"]:
        raise ValueError("Frozen configuration changed")
    for path, expected in preflight["source_hashes"].items():
        if digest(path) != expected:
            raise ValueError(f"Code changed after preflight: {path}")
    _, _, _, model_path = resolve_inputs(config)
    for name, expected in preflight["model_hashes"].items():
        if digest(model_path / name) != expected:
            raise ValueError(f"Pinned model changed: {name}")
    destination = output / args.worker
    destination.mkdir()
    with RssSampler() as rss:
        import torch
        from article_research.hierarchy_packing import (
            HierarchyAwarePackingChunker,
            HierarchyPackingConfig,
        )
        from chunking.fixed import (
            FixedChunkingConfig,
            FixedTokenChunker,
            TokenizerConfig,
        )
        from chunking.structure import StructureAwareChunker, StructureChunkingConfig
        from enrichment.chunks import enrich_chunks
        from retrieval.ranking import (
            BM25Index,
            DenseIndex,
            reciprocal_rank_fusion,
            resolve_text_budget,
        )
        from sentence_transformers import SentenceTransformer

        torch.set_num_threads(config["model"]["cpu_threads"])
        torch.manual_seed(config["seed"])
        np.random.seed(config["seed"])
        documents, sources, gold, _ = load_documents(config, smoke=args.smoke)
        stages = {}
        start = time.perf_counter()
        model = SentenceTransformer(str(model_path), device="cpu", local_files_only=True)
        stages["model_load_seconds"] = time.perf_counter() - start
        budget = resolve_text_budget(model, requested_max_tokens=config["model"]["requested_max_tokens"])
        if budget < 480:
            raise ValueError("Encoder cannot support the frozen target")
        tokenizer_config = TokenizerConfig(name=config["model"]["name"], revision=config["model"]["revision"])
        parameters = config["variants"][args.worker]
        if args.worker == "fixed":
            chunker = FixedTokenChunker(FixedChunkingConfig(tokenizer=tokenizer_config, **parameters), tokenizer=model.tokenizer)
        elif args.worker == "structure":
            chunker = StructureAwareChunker(StructureChunkingConfig(
                tokenizer=tokenizer_config, max_tokens=budget, **parameters), tokenizer=model.tokenizer)
        else:
            chunker = HierarchyAwarePackingChunker(HierarchyPackingConfig(
                tokenizer=tokenizer_config, max_tokens=budget, **parameters), tokenizer=model.tokenizer)
        start = time.perf_counter()
        by_doc, diagnostics = {}, []
        for doc in documents:
            by_doc[doc.doc_id] = chunker.chunk(doc)
            diagnostics.extend({"doc_id": doc.doc_id, **entry} for entry in getattr(chunker, "diagnostics", []))
        stages["segmentation_seconds"] = time.perf_counter() - start
        start = time.perf_counter()
        validation = [validate_chunks(doc, by_doc[doc.doc_id], model.tokenizer) for doc in documents]
        chunks = [chunk for doc in documents for chunk in enrich_chunks(by_doc[doc.doc_id], doc, sources[doc.doc_id])]
        if any(chunk.token_count > budget for chunk in chunks):
            raise ValueError("Encoder text-budget violation")
        stages["integrity_validation_seconds"] = time.perf_counter() - start
        pq.write_table(pa.Table.from_pylist([chunk.model_dump(mode="json") for chunk in chunks]),
                       destination / "chunks.parquet", compression="zstd")
        (destination / "heading_diagnostics.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in diagnostics), encoding="utf-8")
        start = time.perf_counter()
        bm25 = BM25Index(chunks)
        stages["bm25_build_seconds"] = time.perf_counter() - start
        start = time.perf_counter()
        dense = DenseIndex(chunks, model, batch_size=config["model"]["batch_size"])
        stages["dense_build_including_encoder_validation_seconds"] = time.perf_counter() - start
        print(f"{args.worker}: index built; {len(chunks)} chunks; budget={budget}", flush=True)
        chunk_index = {chunk.chunk_id: index for index, chunk in enumerate(chunks)}
        rank_schema = pa.schema([
            ("query_id", pa.string()), ("method", pa.string()), ("rank", pa.int32()),
            ("chunk_index", pa.int32()), ("score", pa.float64()),
        ])
        metrics = []
        start = time.perf_counter()
        with pq.ParquetWriter(destination / "rankings.parquet", rank_schema, compression="zstd") as writer, \
                (destination / "contexts.jsonl").open("w", encoding="utf-8") as context_file:
            for index, question in enumerate(gold, 1):
                question_text = question["question"]
                sparse, semantic = bm25.search(question_text), dense.search(question_text)
                hybrid = reciprocal_rank_fusion(sparse, semantic, constant=config["retrieval"]["fusion_constant"],
                                               depth=config["retrieval"]["fusion_depth"])
                for method, ranking in [("bm25", sparse), ("dense", semantic), ("hybrid", hybrid)]:
                    spans = [(s["char_start"], s["char_end"]) for s in question["gold_spans"]]
                    metrics.append({
                        "variant": args.worker, "method": method, "query_id": question["question_id"],
                        "gold_doc_id": question["doc_id"],
                        **evidence_metrics(ranking, question["doc_id"], spans),
                        **document_metrics(ranking, [question["doc_id"]]),
                    })
                    selected, used = select_ranked_context(ranking, 1024)
                    context_file.write(json.dumps({
                        "query_id": question["question_id"], "method": method, "used_tokens": used,
                        "chunks": [{"chunk_id": item.chunk.chunk_id, "doc_id": item.chunk.doc_id,
                                    "char_start": item.chunk.char_start, "char_end": item.chunk.char_end,
                                    "token_count": item.chunk.token_count, "rank": item.rank} for item in selected],
                    }) + "\n")
                    n = len(ranking)
                    writer.write_table(pa.Table.from_pydict({
                        "query_id": [question["question_id"]] * n, "method": [method] * n,
                        "rank": [item.rank for item in ranking],
                        "chunk_index": [chunk_index[item.chunk.chunk_id] for item in ranking],
                        "score": [item.score for item in ranking],
                    }, schema=rank_schema))
                if index % 20 == 0 or index == len(gold):
                    print(f"{args.worker}: {index}/{len(gold)} questions", flush=True)
        stages["query_ranking_scoring_and_artifact_io_seconds"] = time.perf_counter() - start
        pq.write_table(pa.Table.from_pylist(metrics), destination / "per_question_metrics.parquet", compression="zstd")
        lengths = [chunk.token_count for chunk in chunks]
        overlaps = []
        for doc in documents:
            right = 0
            for chunk in by_doc[doc.doc_id]:
                end = min(right, chunk.char_end)
                overlaps.append(len(model.tokenizer(doc.canonical_text[chunk.char_start:end],
                                add_special_tokens=False, truncation=False)["input_ids"]) if end > chunk.char_start else 0)
                right = max(right, chunk.char_end)
        stats = {
            "documents": len(documents), "questions": len(gold), "chunk_count": len(chunks),
            "indexed_tokens": sum(lengths), "one_chunk_documents": sum(len(c) == 1 for c in by_doc.values()),
            "chunk_tokens": {"min": min(lengths), "p10": float(np.quantile(lengths, .1)),
                             "median": float(np.median(lengths)), "p90": float(np.quantile(lengths, .9)), "max": max(lengths)},
            "actual_overlap_tokens_sum": sum(overlaps), "actual_overlap_tokens_max": max(overlaps),
            "fallback_chunks": sum(chunk.fallback_split for chunk in chunks),
            "encoder_text_budget": budget, "dense_matrix_bytes": dense.vectors.nbytes,
            "dense_matrix_shape": list(dense.vectors.shape), "dense_matrix_dtype": str(dense.vectors.dtype),
            "uncovered_characters": sum(row["uncovered_characters"] for row in validation),
            "uncovered_nonwhitespace_characters": sum(row["uncovered_nonwhitespace_characters"] for row in validation),
            "heading_diagnostic_reasons": dict(Counter(row["reason"] for row in diagnostics)),
            "stages_seconds": stages, "chunker_provenance": chunker.provenance,
            "pid": os.getpid(), "rss_sampling_seconds": .1,
        }
    stats["peak_process_rss_bytes"] = rss.peak
    stats["rss_scope"] = "isolated_variant_process_including_model_and_index_sampled_100ms"
    write_json(destination / "index_statistics.json", stats)
    write_json(destination / "manifest.json", {
        "status": "completed", "finished_at_utc": now(), "variant": args.worker,
        "config_sha256": preflight["config_sha256"], "question_count": len(gold),
        "artifact_sha256": {p.name: digest(p) for p in sorted(destination.iterdir()) if p.is_file()},
    })


def summarize(output, config, documents, gold):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from article_research.packing_evaluation import (
        METRICS,
        document_clusters,
        paired_cluster_bootstrap,
    )

    rows = [row for variant in config["variants"]
            for row in pq.read_table(output / variant / "per_question_metrics.parquet").to_pylist()]
    expected = {(variant, method, q["question_id"]) for variant in config["variants"]
                for method in config["retrieval"]["methods"] for q in gold}
    if len(rows) != len(expected) or {(r["variant"], r["method"], r["query_id"]) for r in rows} != expected:
        raise ValueError("Incomplete paired matrix")
    pq.write_table(pa.Table.from_pylist(rows), output / "per_question_metrics.parquet", compression="zstd")
    aggregates = []
    for variant in config["variants"]:
        for method in config["retrieval"]["methods"]:
            selected = [r for r in rows if r["variant"] == variant and r["method"] == method]
            aggregates.append({"variant": variant, "method": method, "questions": len(selected),
                               **{metric: float(np.mean([r[metric] for r in selected])) for metric in METRICS},
                               "mean_actual_context_tokens": float(np.mean([r["actual_context_tokens_B1024"] for r in selected]))})
    clusters = document_clusters({q["question_id"]: [q["doc_id"]] for q in gold})
    intervals = []
    for reference in ["structure", "fixed", "unrestricted"]:
        for method in config["retrieval"]["methods"]:
            for metric in METRICS:
                def values(variant):
                    return {r["query_id"]: r[metric] for r in rows if r["variant"] == variant and r["method"] == method}
                intervals.append({"contrast": f"hierarchy_minus_{reference}", "method": method, "metric": metric,
                                  **paired_cluster_bootstrap(values(reference), values("hierarchy"), clusters,
                                      repetitions=config["evaluation"]["bootstrap_repetitions"],
                                      seed=config["evaluation"]["bootstrap_seed"])})
    spans_by_variant = {}
    for variant in ["hierarchy", "unrestricted"]:
        spans = defaultdict(list)
        for row in pq.read_table(output / variant / "chunks.parquet", columns=["doc_id", "char_start", "char_end"]).to_pylist():
            spans[row["doc_id"]].append((row["char_start"], row["char_end"]))
        spans_by_variant[variant] = spans
    equal = [doc.doc_id for doc in documents if spans_by_variant["hierarchy"][doc.doc_id] == spans_by_variant["unrestricted"][doc.doc_id]]
    deep = [doc.doc_id for doc in documents if any(b.type == "heading" and b.level >= 3 for b in doc.blocks)]
    shallow = [doc.doc_id for doc in documents if doc.doc_id not in set(deep)]
    equality = {
        "all_documents": {"count": len(documents), "identical": len(equal), "fraction": len(equal) / len(documents)},
        "h3_or_deeper": {"count": len(deep), "identical": len(set(equal) & set(deep)),
                         "fraction": len(set(equal) & set(deep)) / len(deep) if deep else None},
        "no_h3_or_deeper": {"count": len(shallow), "identical": len(set(equal) & set(shallow)),
                            "fraction": len(set(equal) & set(shallow)) / len(shallow) if shallow else None},
    }
    summary = {
        "status": "completed_smoke" if len(gold) != 118 else "completed_exploratory_exposed_ai_labels",
        "question_count": len(gold), "cluster_count": len(set(clusters.values())),
        "corpus_documents": len(documents), "per_question_rows": len(rows),
        "aggregates": aggregates, "paired_intervals": intervals, "ablation_span_equality": equality,
        "costs": {variant: read_json(output / variant / "index_statistics.json") for variant in config["variants"]},
    }
    write_json(output / "summary.json", summary)
    write_json(output / "document_clusters.json", clusters)
    import csv
    with (output / "aggregate_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregates[0]))
        writer.writeheader()
        writer.writerows(aggregates)
    with (output / "paired_intervals.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(intervals[0]))
        writer.writeheader()
        writer.writerows(intervals)
    write_reports(output, summary)
    return summary


def write_reports(output, summary):
    aggregates, costs = summary["aggregates"], summary["costs"]
    lookup = {(r["variant"], r["method"]): r for r in aggregates}
    primary = next(r for r in summary["paired_intervals"] if r["contrast"] == "hierarchy_minus_structure"
                   and r["method"] == "hybrid" and r["metric"] == "EvidenceCoverage@B=1024")
    h, s = lookup[("hierarchy", "hybrid")], lookup[("structure", "hybrid")]
    lines = [
        "# Исследовательское сравнение HierarchyAwarePackingChunker", "",
        f"Статус: {summary['status']}. {summary['question_count']} вопросов; {summary['cluster_count']} документных кластеров; "
        f"{summary['corpus_documents']} документов в поисковом корпусе. Разметка AI-only, выборка уже просмотрена. "
        "Это новые измерения текущей реализации; независимое подтверждение и точное воспроизведение исторической среды не заявляются.", "",
        "## Основной результат", "",
        f"Hybrid EvidenceCoverage@B=1024: hierarchy {h['EvidenceCoverage@B=1024']:.6f}, "
        f"structure {s['EvidenceCoverage@B=1024']:.6f}; разность {primary['difference']:+.6f}, "
        f"описательный 95% кластерный интервал [{primary['ci_lower']:+.6f}; {primary['ci_upper']:+.6f}].", "",
        "## Агрегированные показатели", "",
        "| Метод | Retriever | Coverage B1024 | Coverage@5 | EvidenceRecall@5 | EvidenceMRR@5 | DocumentRecall@5 | DocumentMRR@10 | Средние токены контекста |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregates:
        lines.append(f"| {row['variant']} | {row['method']} | " + " | ".join(f"{row[key]:.4f}" for key in
            ["EvidenceCoverage@B=1024", "EvidenceCoverage@5", "EvidenceRecall@5", "EvidenceMRR@5", "DocumentRecall@5", "DocumentMRR@10"])
            + f" | {row['mean_actual_context_tokens']:.1f} |")
    lines += ["", "## Парные сравнения основной метрики hybrid", "",
              "| Разность | Оценка | Описательный 95% интервал |", "|---|---:|---|" ]
    for row in summary["paired_intervals"]:
        if row["method"] == "hybrid" and row["metric"] == "EvidenceCoverage@B=1024":
            lines.append(f"| {row['contrast']} | {row['difference']:+.6f} | [{row['ci_lower']:+.6f}; {row['ci_upper']:+.6f}] |")
    lines += ["", "## Размеры и затраты", "",
              "| Метод | Чанки | Индексируемые токены | P10 / медиана / P90 | Фактический overlap, токены | Сегментация, с | BM25, с | Dense с проверкой, с | Пик RSS, MiB | Dense-матрица, MiB |",
              "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|"]
    for variant, row in costs.items():
        length, stage = row["chunk_tokens"], row["stages_seconds"]
        lines.append(f"| {variant} | {row['chunk_count']} | {row['indexed_tokens']} | "
            f"{length['p10']:.0f} / {length['median']:.0f} / {length['p90']:.0f} | {row['actual_overlap_tokens_sum']} | "
            f"{stage['segmentation_seconds']:.2f} | {stage['bm25_build_seconds']:.2f} | "
            f"{stage['dense_build_including_encoder_validation_seconds']:.2f} | "
            f"{row['peak_process_rss_bytes']/2**20:.1f} | {row['dense_matrix_bytes']/2**20:.2f} |")
    lines += ["", "Каждый вариант запущен в отдельном процессе. RSS — наблюдавшийся пик всего процесса, "
              "включая модель и индекс, с опросом каждые 100 мс. Измерение времени однократное; файловые кэши ОС "
              "между процессами не очищались. Dense-матрица не является полным индексом.", "", "## Совпадение с абляцией", ""]
    for group, row in summary["ablation_span_equality"].items():
        lines.append(f"- {group}: одинаковые spans у {row['identical']} из {row['count']} документов.")
    lines += ["", "## Определения и ограничения", "",
        "- Основной endpoint и контраст зафиксированы до retrieval. Соседние H2 под одним реальным H1 разрешены; "
        "разные корни разделены. Общий родитель — структурный признак, не доказательство смысловой зависимости.",
        "- EvidenceCoverage — доля покрытых canonical-символов всех непересекающихся gold spans. "
        "Перекрытия чанков внутри gold считаются один раз. Усреднение по вопросам невзвешенное.",
        "- EvidenceRecall@5 = 1, если объединение top-5 покрывает не менее 0,8 всего gold. "
        "EvidenceMRR@5 — обратный ранг первого чанка в top-5, который один покрывает не менее 0,8 всего gold, иначе 0.",
        "- DocumentRecall@5 и DocumentMRR@10 считаются отдельно после max-pool по документам до top-k. "
        "BM25/dense используют полную рассчитанную выдачу, hybrid — всех кандидатов прежнего RRF depth=100. "
        "Эта агрегация не меняет evidence-контекст.",
        "- Контекст B1024: целые чанки по рангу, непомещающиеся пропускаются, усечения нет. "
        "Повторный overlap оплачивается повторно; специальные токены encoder не входят в прежний бюджет возвращаемого текста.",
        f"- Парный bootstrap: {summary['cluster_count']} документных кластеров, 10 000 повторов, seed=2026. "
        "Вопросы с общими gold-документами объединяются транзитивно. Интервалы описательные, без поправки на множественность.",
        "- Parser и canonical text сохранены. HTML-комментарии попадают в encoder и расходуют токены; "
        "таблицы и код не выделяются отдельными типами Markdown-блоков. Gold оценивается относительно canonical text, не сырого Markdown.",
        "- Эксперимент проверяет структурную упаковку. Он не оценивает автоматическое распознавание смысловой зависимости "
        "или качество генерации ответов. Уменьшение индекса само по себе не доказывает улучшение retrieval.",
        "- 118 вопросов ранее просмотрены и размечены AI; исходные 30 test ранее анализировались и здесь не используются. "
        "Переносимость на независимую выборку, другие корпуса и модели не установлена.",
        "", "Подробные результаты по вопросам: per_question_metrics.parquet; агрегаты: aggregate_metrics.csv; "
        "интервалы: paired_intervals.csv. В каталогах вариантов — полные рассчитанные rankings, контекст и chunks. "
        "Связь rankings с chunks задаёт нулевой chunk_index в порядке строк chunks.parquet.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    interpretation = ("Интервал основной разности включает ноль, поэтому направленный выигрыш в этом сравнении не установлен."
                      if primary["ci_lower"] <= 0 <= primary["ci_upper"] else
                      "В данном исследовательском сравнении интервал основной разности не включает ноль; "
                      "это не устраняет ограничения ранее просмотренной AI-разметки.")
    text = (
        "# Текст для раздела результатов статьи\n\n"
        "Проведено контролируемое сравнение четырёх способов сегментации Markdown-представления watsonxDocsQA: "
        "fixed 480/20, исходного structure-aware, предложенной упаковки с ограничением по непосредственному родителю "
        "и её абляции без структурных ограничений. Парсер, канонизация, модель и алгоритмы поиска сохранены. "
        f"В поисковый корпус включены {summary['corpus_documents']} документа; оценка выполнена на "
        f"{summary['question_count']} ранее просмотренных вопросах с AI-разметкой. Параметры зафиксированы до запуска.\n\n"
        "Контекст определяется структурным окружением блока. Соседние подразделы с общим непосредственным родителем "
        "могут объединяться при целевом размере 480 токенов и фактическом encoder budget. Общий родитель "
        "не интерпретируется как установленная смысловая зависимость.\n\n"
        f"Для hybrid RRF основная метрика EvidenceCoverage при бюджете 1024 токена составила "
        f"{h['EvidenceCoverage@B=1024']:.4f} у нового метода и {s['EvidenceCoverage@B=1024']:.4f} у structure-aware. "
        f"Парная разность равна {primary['difference']:+.4f}; описательный 95%-й кластерный bootstrap-интервал "
        f"составил [{primary['ci_lower']:+.4f}; {primary['ci_upper']:+.4f}]. {interpretation}\n\n"
        f"Новый метод создал {costs['hierarchy']['chunk_count']} чанков против {costs['structure']['chunk_count']} "
        f"у structure-aware. Совпадение границ нового метода и абляции наблюдалось в "
        f"{summary['ablation_span_equality']['all_documents']['identical']} документах. "
        "Изменение размера индекса рассматривается совместно с качеством, без отождествления меньшего индекса с лучшим поиском.\n\n"
        "Результаты имеют исследовательский характер. Выборка уже использовалась в анализе, разметка выполнена AI. "
        "Оценка отражает покрытие зафиксированных evidence spans, а не достаточность ответа или качество генерации. "
        "Представление сохраняет комментарии и плоскую обработку таблиц и кода; эти ограничения общие для всех методов.\n"
    )
    (output / "article_results.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ARTICLE / "configs/hierarchy_packing.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--worker", choices=["fixed", "structure", "hierarchy", "unrestricted"])
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    from run_stage_a_baseline import environment, validate_run_dir
    output = validate_run_dir(args.output_dir, ARTICLE)
    config = read_json(args.config)
    snapshot, corpus, gold_path, model = resolve_inputs(config)
    # Public config has only one ablation difference, locked before any scores.
    h, u = config["variants"]["hierarchy"], config["variants"]["unrestricted"]
    if {k: v for k, v in h.items() if k != "respect_parent_boundary"} != {k: v for k, v in u.items() if k != "respect_parent_boundary"}:
        raise ValueError("Ablation differs beyond hierarchy flag")
    documents, _, gold, raw_rows = load_documents(config, smoke=args.smoke)
    output.mkdir(parents=True)
    write_json(output / "frozen_config.json", config)
    source_files = [Path(__file__), ARTICLE / "scripts/prepare_baseline_snapshot.py", ARTICLE / "scripts/run_stage_a_baseline.py",
                    ARTICLE / "docs/hierarchy_packing_protocol.md", ARTICLE / "tests/test_hierarchy_packing.py",
                    *sorted((ARTICLE / "src/article_research").glob("*.py"))]
    source_hashes = {str(p.resolve()): digest(p) for p in source_files}
    model_hashes = {str(p.relative_to(model)): digest(p) for p in sorted(model.rglob("*")) if p.is_file()}
    versions = environment()
    versions["packages"].update({name: importlib.metadata.version(name) for name in ["psutil", "pytest", "beautifulsoup4"]})
    (output / "requirements.lock.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True), encoding="utf-8")
    audit = {"documents": len(documents), "heading_levels": dict(Counter(b.level for d in documents for b in d.blocks if b.type == "heading")),
             "parsed_block_types": dict(Counter(b.type for d in documents for b in d.blocks)),
             "documents_with_comment_blocks": sum(any("<!--" in b.text for b in d.blocks) for d in documents),
             "no_heading_documents": sum(not any(b.type == "heading" for b in d.blocks) for d in documents),
             "one_heading_documents": sum(sum(b.type == "heading" for b in d.blocks) == 1 for d in documents),
             "gold_coordinate_system": "pipeline_document.canonical_text", "gold_exact_span_checks_passed": True}
    if raw_rows:
        from bs4 import BeautifulSoup
        tags, present = Counter(), Counter()
        for row in raw_rows:
            soup = BeautifulSoup(row["html_document"], "html.parser")
            for tag in soup.find_all(["nav", "script", "style", "footer"]):
                tag.decompose()
            for name in ["table", "pre", "code"]:
                count = len(soup.find_all(name))
                tags[name] += count
                present[name] += count > 0
        audit.update({"source_html_tags": dict(tags), "source_html_documents_with_tag": dict(present)})
    write_json(output / "representation_audit.json", audit)
    git_states = {}
    for label, directory in [("workspace", ARTICLE.parent), ("upstream", ARTICLE.parent / "techdoc-chunking-pipeline")]:
        try:
            base = ["git", "-c", f"safe.directory={directory}", "-c", "core.quotepath=false", "-C", str(directory)]
            head = subprocess.run([*base, "rev-parse", "HEAD"], capture_output=True, text=True)
            status = subprocess.run([*base, "status", "--short", "--untracked-files=no"], capture_output=True, text=True)
            git_states[label] = {"head": head.stdout.strip(), "tracked_status": status.stdout.splitlines(),
                                 "returncodes": [head.returncode, status.returncode],
                                 "new_article_sources_identified_by_content_hashes": True}
        except FileNotFoundError:
            git_states[label] = {"available": False, "source_identity": "recorded_content_hashes"}
    write_json(output / "preflight.json", {
        "status": "passed_before_retrieval", "frozen_at_utc": now(), "smoke": args.smoke,
        "config_sha256": digest(output / "frozen_config.json"), "source_hashes": source_hashes,
        "model_hashes": model_hashes, "corpus_path": str(corpus), "corpus_sha256": digest(corpus),
        "gold_path": str(gold_path), "gold_sha256": digest(gold_path),
        "snapshot_manifest_sha256": digest(snapshot / "source_snapshot_manifest.json"),
        "snapshot_id": snapshot.name, "environment": versions,
        "exact_code_state": "recorded_content_hashes_and_available_git_metadata",
        "git_states": git_states,
        "parent_platform": platform.platform(), "command": sys.argv,
    })
    for variant in config["variants"]:
        command = [sys.executable, str(Path(__file__).resolve()), "--output-dir", str(output), "--worker", variant]
        if args.smoke:
            command.append("--smoke")
        result = subprocess.run(command, check=False, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        if result.returncode:
            write_json(output / "failure.json", {"variant": variant, "returncode": result.returncode, "time": now()})
            raise RuntimeError(f"Variant failed; no results substituted: {variant}")
    summary = summarize(output, config, documents, gold)
    write_json(output / "manifest.json", {
        "status": summary["status"], "finished_at_utc": now(),
        "question_count": len(gold), "cluster_count": summary["cluster_count"],
        "artifact_sha256": {str(p.relative_to(output)): digest(p) for p in sorted(output.rglob("*")) if p.is_file()},
    })
    print(json.dumps({"output": str(output), "status": summary["status"], "rows": summary["per_question_rows"]}), flush=True)


if __name__ == "__main__":
    main()
