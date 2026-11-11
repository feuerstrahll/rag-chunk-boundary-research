"""Post-outcome editorial and structural analysis; no indexing or encoder calls.

Reads immutable completed-run artifacts, reconstructs the same canonical documents,
and uses only the pinned fast tokenizer. Does not execute chunk() or search().
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"})
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.dont_write_bytecode = True
ARTICLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ARTICLE / "src"))

from run_hierarchy_packing import (
    digest,
    load_documents,
    read_json,
    resolve_inputs,
    write_json,
)

LENGTH_BANDS = ("<=510", "511-1024", "1025-2048", ">2048")
PRESENTATION_NAMES = {
    "EvidenceRecall@5": "EvidenceSuccess@5_tau0.8",
    "EvidenceMRR@5": "EvidenceMRR@5_tau0.8_single_chunk",
}


def length_band(tokens):
    if tokens <= 510:
        return LENGTH_BANDS[0]
    if tokens <= 1024:
        return LENGTH_BANDS[1]
    if tokens <= 2048:
        return LENGTH_BANDS[2]
    return LENGTH_BANDS[3]


def visible_body(text):
    """Audit-only operational content flag, not a semantic annotation or parser fix."""
    return bool(re.sub(r"<!--(?:.*?-->|.*$)", "", text, flags=re.S).strip())


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def inspect_headings(documents, sources, tokenizer, diagnostics, saved, question_ids_by_doc):
    from article_research.hierarchy_packing import (
        HierarchyAwarePackingChunker,
        HierarchyPackingConfig,
    )
    from chunking.fixed import TokenizerConfig

    # Reconstruct the frozen structural regions only; never generate new chunks.
    chunker = HierarchyAwarePackingChunker(HierarchyPackingConfig(
        tokenizer=TokenizerConfig(name="pinned-audit-tokenizer", revision="audit"),
        target_tokens=480, max_tokens=510, overlap_tokens=64,
        respect_parent_boundary=True,
    ), tokenizer=tokenizer)
    docs = {d.doc_id: d for d in documents}
    regions = {doc_id: chunker._regions(docs[doc_id]) for doc_id in {r["doc_id"] for r in diagnostics}}
    records = []
    for diagnostic in diagnostics:
        document = docs[diagnostic["doc_id"]]
        blocks, text = document.blocks, document.canonical_text
        block_positions = {b.block_id: i for i, b in enumerate(blocks)}
        region = next(r for r in regions[document.doc_id] if any(b.block_id == diagnostic["block_id"] for b in r))
        first = next(i for i, b in enumerate(region) if b.block_id == diagnostic["block_id"])
        pending = region[first:]
        assert pending and all(b.type == "heading" for b in pending)
        heading = pending[-1]
        last_index = block_positions[heading.block_id]
        next_heading = blocks[last_index + 1]
        assert next_heading.type == "heading"
        # Direct body ends at the next heading; subtree includes descendant sections.
        subtree_end = next((i for i in range(last_index + 1, len(blocks))
                            if blocks[i].type == "heading" and blocks[i].level <= heading.level), len(blocks))
        subtree_body = [b for b in blocks[last_index + 1:subtree_end] if b.type != "heading"]
        meaningful = [b for b in subtree_body if visible_body(b.text)]
        body = meaningful[0] if meaningful else None
        heading_chunks = [c for c in saved["hierarchy"][document.doc_id]
                          if c["char_start"] <= heading.char_start and c["char_end"] >= heading.char_end]
        assert heading_chunks
        terminal_chunks = [c for c in heading_chunks if c["char_end"] == heading.char_end]
        assert terminal_chunks, "Diagnostic does not correspond to a saved terminal heading"
        chunk = terminal_chunks[-1]
        in_chunk = [b for b in blocks if b.char_end > chunk["char_start"] and b.char_start < chunk["char_end"]]
        nonheading = [b for b in in_chunk if b.type != "heading"]
        other_body = [b for b in nonheading if visible_body(
            text[max(b.char_start, chunk["char_start"]):min(b.char_end, chunk["char_end"])])]
        following = next(c for c in saved["hierarchy"][document.doc_id] if c["char_start"] >= heading.char_end)

        def count(a, b):
            return len(tokenizer(text[a:b], add_special_tokens=False, truncation=False)["input_ids"])
        full_tokens = count(pending[0].char_start, body.char_end) if body else None
        minimal_tokens = None
        if body:
            offsets = tokenizer(body.text, add_special_tokens=False, truncation=False,
                                return_offsets_mapping=True)["offset_mapping"]
            first_token_end = next(b for a, b in offsets if b > a)
            minimal_tokens = count(pending[0].char_start, body.char_start + first_token_end)
        ablation_with_body = [c for c in saved["unrestricted"][document.doc_id] if body
                              and c["char_start"] <= heading.char_start and c["char_end"] >= body.char_end]
        records.append({
            "doc_id": document.doc_id, "title": sources[document.doc_id].metadata.get("title"),
            "source_url": document.source, "diagnostic_block_id": diagnostic["block_id"],
            "heading_run": [{"block_id": b.block_id, "text": b.text, "level": b.level,
                             "char_start": b.char_start, "char_end": b.char_end} for b in pending],
            "terminal_heading": heading.text, "terminal_heading_level": heading.level,
            "next_heading": next_heading.text, "next_heading_level": next_heading.level,
            "next_heading_is_descendant": next_heading.level > heading.level,
            "direct_body_blocks_before_next_heading": 0,
            "subtree_nonheading_blocks": len(subtree_body), "subtree_body_after_comment_exclusion": len(meaningful),
            "heading_in_separate_chunk_from_first_subtree_body": body is not None and chunk["char_end"] <= body.char_start,
            "heading_chunk_contains_only_headings": not nonheading,
            "heading_chunk_has_other_body_after_comment_exclusion": bool(other_body),
            "heading_chunk_comment_only_nonheading_blocks": len(nonheading) - len(other_body),
            "heading_chunk": chunk,
            "following_saved_chunk": following,
            "first_subtree_body": {"block_id": body.block_id, "char_start": body.char_start,
                                   "char_end": body.char_end, "text": body.text} if body else None,
            "heading_run_through_first_body_tokens": full_tokens,
            "heading_run_through_first_body_token_prefix_tokens": minimal_tokens,
            "full_heading_body_span_fits_target": full_tokens <= 480 if full_tokens is not None else None,
            "full_heading_body_span_fits_encoder": full_tokens <= 510 if full_tokens is not None else None,
            "minimal_heading_body_prefix_fits_encoder": minimal_tokens <= 510 if minimal_tokens is not None else None,
            "move_heading_run_to_following_saved_chunk_tokens": count(pending[0].char_start, following["char_end"]),
            "merge_two_saved_chunks_tokens": count(chunk["char_start"], following["char_end"]),
            "ablation_contains_heading_and_full_first_body": bool(ablation_with_body),
            "ablation_shared_chunk_ids": [c["chunk_id"] for c in ablation_with_body],
            "evaluation_query_ids_in_document": question_ids_by_doc.get(document.doc_id, []),
        })
    with_body = [r for r in records if r["first_subtree_body"] is not None]
    summary = {
        "diagnostic_records": len(records), "documents": len({r["doc_id"] for r in records}),
        "distinct_terminal_heading_chunks": len({r["heading_chunk"]["chunk_id"] for r in records}),
        "heading_runs_with_multiple_headings": sum(len(r["heading_run"]) > 1 for r in records),
        "direct_body_before_next_heading": 0,
        "next_heading_is_descendant": sum(r["next_heading_is_descendant"] for r in records),
        "has_body_in_subtree_after_comment_exclusion": len(with_body),
        "no_body_in_subtree_after_comment_exclusion": len(records) - len(with_body),
        "heading_only_chunk_records": sum(r["heading_chunk_contains_only_headings"] for r in records),
        "chunk_has_other_body_records": sum(r["heading_chunk_has_other_body_after_comment_exclusion"] for r in records),
        "chunk_has_only_headings_and_comment_blocks_records": sum(
            not r["heading_chunk_contains_only_headings"] and not r["heading_chunk_has_other_body_after_comment_exclusion"]
            for r in records),
        "heading_separate_from_first_subtree_body": sum(r["heading_in_separate_chunk_from_first_subtree_body"] for r in records),
        "full_heading_body_fits_target": sum(r["full_heading_body_span_fits_target"] for r in with_body),
        "full_heading_body_fits_encoder": sum(r["full_heading_body_span_fits_encoder"] for r in with_body),
        "full_heading_body_exceeds_encoder": sum(not r["full_heading_body_span_fits_encoder"] for r in with_body),
        "minimal_heading_body_prefix_fits_encoder": sum(r["minimal_heading_body_prefix_fits_encoder"] for r in with_body),
        "transfer_to_following_saved_chunk_fits_encoder": sum(r["move_heading_run_to_following_saved_chunk_tokens"] <= 510 for r in records),
        "whole_saved_chunk_pair_fits_encoder": sum(r["merge_two_saved_chunks_tokens"] <= 510 for r in records),
        "ablation_heading_and_full_first_body_together": sum(r["ablation_contains_heading_and_full_first_body"] for r in with_body),
        "evaluation_questions_in_affected_documents": len({q for r in records for q in r["evaluation_query_ids_in_document"]}),
    }
    assert len(records) == 133
    return records, summary


def render(output, summary, heading_records, grouped):
    a = summary["heading_audit"]
    lines = ["# Дополнительный анализ после основного эксперимента", "",
        "Статус: post-outcome exploratory. Алгоритм, RRF и основной запуск не изменялись. "
        "Нового индексирования, поиска или запуска encoder нет; использованы сохранённые результаты и pinned tokenizer.", "",
        "## Терминология", "",
        "EvidenceSuccess@5, τ=0,8 — доля вопросов, у которых объединение top-5 покрывает не менее 80% gold. "
        "В исходном машинном артефакте этот показатель называется EvidenceRecall@5; значения не изменены. "
        "EvidenceMRR@5, τ=0,8, один чанк — средний обратный ранг первого чанка в top-5, который один покрывает "
        "не менее 80% всего gold вопроса; при отсутствии такого чанка — 0. Документные Recall@5 и MRR@10 остаются отдельными метриками.", "",
        "## 133 диагностические записи о заголовках", "",
        f"Записи относятся к {a['documents']} документам и {a['distinct_terminal_heading_chunks']} различным чанкам. "
        "Это не число доказанных retrieval-ошибок. Для аудита содержимым считается непустой не-heading блок "
        "после удаления HTML-комментариев только в диагностической проверке; текст pipeline не изменён.", "",
        "| Проверка | Число записей |", "|---|---:|",
    ]
    labels = {
        "heading_only_chunk_records": "Чанк содержит только заголовки",
        "chunk_has_other_body_records": "В чанке есть другой текст после исключения комментариев",
        "chunk_has_only_headings_and_comment_blocks_records": "В чанке заголовок и только комментарии вместо другого текста",
        "next_heading_is_descendant": "Следующий заголовок — более глубокого уровня",
        "has_body_in_subtree_after_comment_exclusion": "В поддереве найден последующий текст",
        "no_body_in_subtree_after_comment_exclusion": "В поддереве нет текста после исключения комментариев",
        "heading_separate_from_first_subtree_body": "Первый текст поддерева отделён от завершающего заголовка",
        "full_heading_body_fits_target": "Связка заголовков до конца первого текста помещается в 480",
        "full_heading_body_fits_encoder": "Та же связка помещается в 510",
        "full_heading_body_exceeds_encoder": "Для всей связки дополнительно мешает предел 510",
        "minimal_heading_body_prefix_fits_encoder": "Связка до первого токена текста помещается в 510",
        "transfer_to_following_saved_chunk_fits_encoder": "Заголовки плюс весь следующий сохранённый чанк помещаются в 510",
        "whole_saved_chunk_pair_fits_encoder": "Два сохранённых чанка целиком помещаются в 510",
        "ablation_heading_and_full_first_body_together": "Абляция сохраняет заголовок с полным первым текстовым блоком",
    }
    lines += [f"| {label} | {a[key]} |" for key, label in labels.items()]
    lines += ["", "Во всех случаях после завершающего заголовка до следующего заголовка нет прямого "
        "текстового блока. Поэтому отделённое содержимое следует проверять как содержимое подраздела/поддерева, "
        "а не автоматически считать потерянным вводным абзацем. В 132 случаях с текстом поддерева связка "
        "до конца первого текстового блока помещается даже в целевые 480 токенов; размер не объясняет "
        "это отделение. Во всех 132 случаях в абляции заголовок и полный первый текстовый блок находятся вместе. "
        "Один случай — пустой корневой раздел перед другим H1; требовать связывания с чужим разделом нельзя. "
        "Смена непосредственного родителя действительно "
        "завершает область упаковки; таблица отдельно показывает, когда размер также препятствует объединению. "
        "Недостаток правила: оно может оставить родительский заголовок вне чанка дочернего содержимого даже при "
        "достаточном токенном бюджете. Только один оценочный вопрос относится к этим 23 документам; "
        "это не исключает влияния документов как нерелевантных кандидатов и не устанавливает причинный эффект. "
        "Не установлено, что каждая такая ситуация ухудшает извлечение evidence.", "",
        "### Примеры из сохранённых чанков", ""]
    # Deterministic structural diversity; no selection by retrieval quality.
    predicates = [
        ("Только заголовок; связка с первым текстом помещается", lambda r: r["heading_chunk_contains_only_headings"] and r["full_heading_body_span_fits_encoder"]),
        ("Заголовок в чанке с другим текстом", lambda r: r["heading_chunk_has_other_body_after_comment_exclusion"]),
        ("Объединение полного содержимого ограничено также размером", lambda r: r["first_subtree_body"] and not r["full_heading_body_span_fits_encoder"]),
        ("Несколько завершающих заголовков", lambda r: len(r["heading_run"]) > 1),
        ("Пустое поддерево после исключения комментариев", lambda r: r["first_subtree_body"] is None),
    ]
    selected, seen = [], set()
    for label, predicate in predicates:
        candidate = next((r for r in heading_records if predicate(r) and r["diagnostic_block_id"] not in seen), None)
        if candidate:
            selected.append((label, candidate))
            seen.add(candidate["diagnostic_block_id"])
    for label, record in selected:
        chunk, following = record["heading_chunk"], record["following_saved_chunk"]
        lines += [f"**{label}. {record['title']}**", "",
            f"Документ `{record['doc_id']}`; H{record['terminal_heading_level']} «{record['terminal_heading']}» → "
            f"H{record['next_heading_level']} «{record['next_heading']}». "
            f"Чанк `{chunk['chunk_id']}`, span [{chunk['char_start']}, {chunk['char_end']}), {chunk['token_count']} токенов. "
            f"Связка до конца первого текстового блока: {record['heading_run_through_first_body_tokens'] if record['first_subtree_body'] else 'нет собственного текста поддерева'}; "
            f"перенос в следующий сохранённый чанк: {record['move_heading_run_to_following_saved_chunk_tokens']} токенов.", "",
            "Завершение текущего сохранённого чанка (не более 800 символов):", "",
            *["> " + part for part in chunk["text"][-800:].splitlines()], "",
            f"Начало следующего чанка `{following['chunk_id']}`, span [{following['char_start']}, {following['char_end']}):", "",
            *["> " + part for part in following["text"][:650].splitlines()], ""]
    lines += ["Полные тексты и offsets всех 133 записей сохранены в heading_cases.jsonl. "
              "Примеры выбраны по структурным категориям, без отбора по retrieval-метрикам.", "",
              "## Длины canonical-документов", "",
              "Длины посчитаны закреплённым tokenizer без специальных токенов и без усечения; комментарии включены. "
              "Диапазоны выбраны после основного эксперимента: ≤510, 511–1024, 1025–2048, >2048. "
              "Они соответствуют encoder budget и кратным бюджету возвращаемого контекста. "
              "Документ длиннее 1024 токенов условно считается длинным только для этого дополнительного разреза.", "",
              "| Диапазон, токены | Документы корпуса | Оценочные вопросы |", "|---|---:|---:|"]
    lines += [f"| {b} | {summary['corpus_bands'][b]} | {summary['question_bands'][b]} |" for b in LENGTH_BANDS]
    lines += ["", "Распределение по всему корпусу: " + "; ".join(f"{k}={v:.1f}" for k, v in summary["token_length_distribution"].items()) + ".", "",
              "### Hybrid Coverage@1024 по диапазонам", "",
              "| Длина | N | Fixed | Structure | Hierarchy | Без ограничений | H−S, п.п. | H−F, п.п. | H−U, п.п. |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for band in LENGTH_BANDS:
        values = {r["variant"]: r for r in grouped if r["method"] == "hybrid" and r["length_band"] == band}
        n = summary["question_bands"][band]
        if n:
            v = {name: row["EvidenceCoverage@B=1024"] for name, row in values.items()}
            lines.append(f"| {band} | {n} | {v['fixed']:.4f} | {v['structure']:.4f} | {v['hierarchy']:.4f} | "
                         f"{v['unrestricted']:.4f} | {(v['hierarchy']-v['structure'])*100:+.2f} | "
                         f"{(v['hierarchy']-v['fixed'])*100:+.2f} | {(v['hierarchy']-v['unrestricted'])*100:+.2f} |")
        else:
            lines.append(f"| {band} | 0 | — | — | — | — | — | — | — |")
    lines += ["", "Все retriever представлены в length_group_metrics.csv и length_group_contrasts.csv. "
              "Групповые показатели описательные; дополнительные интервалы и подтверждающие тесты не рассчитывались. "
              "Размеры групп указаны явно; данный разрез не заменяет основной результат на 118 вопросах.", "",
              "## Обсуждение retrieval", "",
              "У hierarchy packing BM25 Coverage@1024 = 0,8044, hybrid = 0,7366, dense = 0,6010. "
              "Гибридный поиск в выбранной конфигурации не обеспечил максимальное покрытие доказательств; "
              "влияние сегментации различалось между поисковыми компонентами. "
              "Dense у hierarchy ниже structure (0,6010 против 0,6095). "
              "Сравнение BM25 с hybrid здесь описательное, основной заранее выбранный retriever остаётся hybrid. "
              "RRF не перенастраивался.", "",
              "## Проверки", "",
              "Имена в aggregate_metrics_presentation.csv изменены только для представления. Значения совпадают "
              "с исходными метриками. Суммы размеров диапазонов равны 1144 документам и 118 вопросам. "
              "Взвешенные по числу вопросов групповые средние воспроизводят исходные 12 агрегатов. "
              "Хеши всех исходных артефактов проверены до и после анализа. Алгоритм не исправлялся задним числом."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run, output = args.run.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError("Posthoc output must be new")
    if not output.is_relative_to(ARTICLE / "results"):
        raise ValueError("Output must stay inside article/results")
    manifest = read_json(run / "manifest.json")
    expected = {run / name: value for name, value in manifest["artifact_sha256"].items()}
    expected[run / "manifest.json"] = digest(run / "manifest.json")
    def check_immutable():
        if any(digest(path) != sha for path, sha in expected.items()):
            raise ValueError("Completed-run artifact changed")
    check_immutable()
    config = read_json(run / "frozen_config.json")
    preflight = read_json(run / "preflight.json")
    for path, value in preflight["source_hashes"].items():
        if digest(path) != value:
            raise ValueError(f"Frozen source changed: {path}")
    _, _, _, model_path = resolve_inputs(config)
    import numpy as np
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, use_fast=True)
    for name, value in preflight["model_hashes"].items():
        if any(word in name for word in ["tokenizer", "vocab", "special_tokens", "config.json"]):
            if digest(model_path / name) != value:
                raise ValueError("Pinned tokenizer changed")
    documents, sources, gold, raw_rows = load_documents(config)
    del raw_rows
    lengths = {d.doc_id: len(tokenizer(d.canonical_text, add_special_tokens=False, truncation=False)["input_ids"])
               for d in documents}
    corpus_bands = {b: sum(length_band(n) == b for n in lengths.values()) for b in LENGTH_BANDS}
    question_bands = {b: sum(length_band(lengths[q["doc_id"]]) == b for q in gold) for b in LENGTH_BANDS}
    assert sum(corpus_bands.values()) == 1144 and sum(question_bands.values()) == 118
    rows = pq.read_table(run / "per_question_metrics.parquet").to_pylist()
    grouped, contrasts = [], []
    for band in LENGTH_BANDS:
        for method in ["bm25", "dense", "hybrid"]:
            means = {}
            for variant in config["variants"]:
                selected = [r for r in rows if r["method"] == method and r["variant"] == variant
                            and length_band(lengths[r["gold_doc_id"]]) == band]
                assert len(selected) == question_bands[band]
                value = float(np.mean([r["EvidenceCoverage@B=1024"] for r in selected])) if selected else None
                means[variant] = value
                grouped.append({"length_band": band, "method": method, "variant": variant,
                                "questions": len(selected), "EvidenceCoverage@B=1024": value})
            for reference in ["structure", "fixed", "unrestricted"]:
                contrasts.append({"length_band": band, "method": method, "contrast": f"hierarchy_minus_{reference}",
                                  "questions": question_bands[band],
                                  "difference": means["hierarchy"] - means[reference] if question_bands[band] else None})
    aggregates = read_json(run / "summary.json")["aggregates"]
    for aggregate in aggregates:
        strata = [r for r in grouped if r["variant"] == aggregate["variant"] and r["method"] == aggregate["method"]]
        reconstructed = sum(r["questions"] * r["EvidenceCoverage@B=1024"] for r in strata if r["questions"]) / 118
        assert math.isclose(reconstructed, aggregate["EvidenceCoverage@B=1024"], abs_tol=1e-12)
    presentation = [{PRESENTATION_NAMES.get(key, key): value for key, value in row.items()} for row in aggregates]
    by_doc = defaultdict(list)
    for q in gold:
        by_doc[q["doc_id"]].append(q["question_id"])
    saved = {}
    for variant in ["hierarchy", "unrestricted"]:
        saved[variant] = defaultdict(list)
        for chunk in pq.read_table(run / variant / "chunks.parquet").to_pylist():
            saved[variant][chunk["doc_id"]].append(chunk)
    diagnostics = [json.loads(line) for line in (run / "hierarchy/heading_diagnostics.jsonl").read_text().splitlines()]
    assert all(r["reason"] == "structural_boundary" for r in diagnostics)
    cases, heading_summary = inspect_headings(documents, sources, tokenizer, diagnostics, saved, by_doc)
    distribution = {"min": min(lengths.values()), **{f"p{p}": float(np.percentile(list(lengths.values()), p))
                    for p in [10, 25, 50, 75, 90, 95]}, "max": max(lengths.values())}
    summary = {"status": "post_outcome_exploratory_no_new_indexing", "corpus_bands": corpus_bands,
               "question_bands": question_bands, "token_length_distribution": distribution,
               "heading_audit": heading_summary, "presentation_metric_mapping": PRESENTATION_NAMES}
    output.mkdir(parents=True)
    write_csv(output / "document_lengths.csv", [{"doc_id": d, "canonical_tokens": n, "length_band": length_band(n),
              "evaluation_questions": len(by_doc[d])} for d, n in lengths.items()])
    write_csv(output / "length_group_metrics.csv", grouped)
    write_csv(output / "length_group_contrasts.csv", contrasts)
    write_csv(output / "aggregate_metrics_presentation.csv", presentation)
    (output / "heading_cases.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in cases), encoding="utf-8")
    write_json(output / "summary.json", summary)
    render(output, summary, cases, grouped)
    check_immutable()
    write_json(output / "manifest.json", {
        "status": summary["status"], "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_run_manifest_sha256": digest(run / "manifest.json"),
        "source_script_sha256": digest(Path(__file__)), "source_artifact_hashes_verified_before_and_after": True,
        "frozen_sources_unchanged": True, "encoder_inference_or_indexing": False,
        "versions": {name: importlib.metadata.version(name) for name in ["transformers", "tokenizers", "numpy", "pyarrow"]},
        "length_bands": list(LENGTH_BANDS), "groups_selected_after_primary_results": True,
        "artifact_sha256": {p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()},
    })
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
