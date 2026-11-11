"""Post-outcome explanation of zero paired coverage and two existing limitations.

Read-only with respect to the completed experiment. No labels, queries, chunking,
encoder inference, retrieval or parameter changes are introduced.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from run_context_packing import check_hashes
from run_hierarchy_packing import ARTICLE, digest, now, read_json, write_json


def main():
    import pyarrow.parquet as pq

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run, output = args.run.resolve(), args.output.resolve()
    if output.exists() or not output.is_relative_to(ARTICLE / "results/analyses"):
        raise ValueError("A new analysis directory under article/results/analyses is required")
    manifest = read_json(run / "manifest.json")
    immutable = {run / name: sha for name, sha in manifest["artifact_sha256"].items()}
    immutable[run / "manifest.json"] = digest(run / "manifest.json")
    check_hashes(immutable)
    config, summary = read_json(run / "frozen_config.json"), read_json(run / "summary.json")
    assert summary["heading_behavior"]["headings"]["known_cases_preserved"] == 132
    assert summary["detector"]["headings_intro"]["intro_links_proposed"] == 6
    assert summary["semantic_label_status"]["evaluation_questions_in_labeled_documents"] == 0
    preflight = read_json(run / "preflight.json")
    check_hashes(preflight["source_hashes"])
    check_hashes(preflight["immutable_source_artifacts"])
    rows = pq.read_table(run / "per_question_metrics.parquet").to_pylist()
    metrics = ["EvidenceCoverage@B=1024", "EvidenceCoverage@5", "EvidenceMRR@5",
               "DocumentMRR@10", "actual_context_tokens_B1024"]
    differences = []
    for method in config["retrieval"]["methods"]:
        a = {r["query_id"]: r for r in rows if r["variant"] == "v1" and r["method"] == method}
        b = {r["query_id"]: r for r in rows if r["variant"] == "headings" and r["method"] == method}
        assert a.keys() == b.keys() and len(a) == 118
        differences.append({"method": method, "paired_queries": len(a),
                            **{name: sum(a[q][name] != b[q][name] for q in a) for name in metrics}})
    assert all(r["EvidenceCoverage@B=1024"] == r["EvidenceCoverage@5"] == 0 for r in differences)
    folders = {"v1": Path(preflight["baseline_run"]) / "hierarchy",
               **{v: run / v for v in config["variants"]}}
    chunks = {v: pq.read_table(p / "chunks.parquet").to_pylist() for v, p in folders.items()}
    spans = {v: defaultdict(list) for v in folders}
    for variant, values in chunks.items():
        for c in values:
            spans[variant][c["doc_id"]].append((c["char_start"], c["char_end"]))
    changed = sorted(d for d in spans["v1"] if spans["v1"][d] != spans["headings"][d])
    gold_path = ARTICLE / config["gold"]["path"]
    if digest(gold_path) != config["gold"]["sha256"]:
        raise ValueError("Frozen gold input changed")
    gold = [json.loads(line) for line in gold_path.read_text().splitlines()]
    changed_queries = [{"query_id": q["question_id"], "doc_id": q["doc_id"]} for q in gold if q["doc_id"] in changed]
    groups = {r["group_id"]: r for r in (json.loads(line) for line in (run / "evaluation_groups.jsonl").read_text().splitlines())}
    missing = [r for r in (json.loads(line) for line in (run / "retrieved_group_details.jsonl").read_text().splitlines())
               if r["method"] == "hybrid" and r["context"] == "B1024" and r["cohort"] == "structural_chain"
               and r["is_gold_document"] and r["b_fully_returned"] and not r["complete"]]
    assert len({(r["query_id"], r["group_id"]) for r in missing}) == 1
    from article_research.context_evaluation import index_preserved, retrieved_state

    examples = []
    for record in missing:
        variant, query_id = record["variant"], record["query_id"]
        group = groups[record["group_id"]]
        context = next(r for r in (json.loads(line) for line in (folders[variant] / "contexts.jsonl").read_text().splitlines())
                       if r["query_id"] == query_id and r["method"] == "hybrid")
        assert index_preserved(group, chunks[variant]) and not retrieved_state(group, context["chunks"])["complete"]
        ranking = pq.read_table(folders[variant] / "rankings.parquet",
                                filters=[("query_id", "==", query_id), ("method", "==", "hybrid")]).to_pylist()
        ranks = {r["chunk_index"]: r["rank"] for r in ranking}
        together = []
        for index, c in enumerate(chunks[variant]):
            if c["doc_id"] == group["doc_id"] and index_preserved(group, [c]):
                rank = ranks.get(index)
                used_before = sum(s["token_count"] for s in context["chunks"] if rank is not None and s["rank"] < rank)
                together.append({"chunk_id": c["chunk_id"], "span": [c["char_start"], c["char_end"]],
                                 "tokens": c["token_count"], "hybrid_rank": rank,
                                 "budget_remaining_at_rank": 1024 - used_before,
                                 "a_quotes": [c["text"][a-c["char_start"]:b-c["char_start"]] for a, b in group["a_spans"]]})
        examples.append({"variant": variant, "query_id": query_id, "group": group,
                         "co_located_index_chunks": together, "selected_context": context})
    with (run / "index_preservation.csv").open() as handle:
        not_handled = [r for r in csv.DictReader(handle) if r["cohort"] == "existing_content_checked_AI"
                       and r["variant"] == "headings_intro" and r["preserved"] == "False"]
    assert len(not_handled) == 1
    for row in not_handled:
        group = groups[row["group_id"]]
        def quote(span, group=group):
            a, b = span
            c = next(c for c in chunks["headings_intro"] if c["doc_id"] == group["doc_id"]
                     and c["char_start"] <= a and c["char_end"] >= b)
            return c["text"][a-c["char_start"]:b-c["char_start"]]
        row["a_quotes"] = [quote(s) for s in group["a_spans"]]
        row["b_quote"] = quote(group["b_span"])
    diagnostics = {"status": "post_outcome_exploratory_interpretation_no_method_changes",
                   "paired_changed_query_counts": differences, "changed_document_ids": changed,
                   "evaluation_queries_in_changed_documents": changed_queries,
                   "index_preserved_but_returned_context_incomplete": examples,
                   "existing_AI_label_outside_rule": not_handled,
                   "retrieval_rerun": False, "new_semantic_labels": 0}
    output.mkdir(parents=True)
    write_json(output / "diagnostics.json", diagnostics)
    table = "\n".join(f"| {r['method']} | {r[metrics[0]]} | {r[metrics[1]]} | {r[metrics[2]]} | {r[metrics[3]]} | {r[metrics[4]]} |"
                      for r in differences)
    example = next(r for r in examples if r["variant"] == "headings")
    co = example["co_located_index_chunks"][0]
    unhandled = not_handled[0]
    text = f"""# Интерпретация результата V2

Статус: post-outcome exploratory. Основной завершённый запуск, его параметры,
код и результаты не изменялись. Нового поиска, encoder inference и разметки нет.
Основной отчёт: [context_full_20261027_v2/report.md](../../runs/context_full_20261027_v2/report.md).

## Почему основной bootstrap-интервал равен [0; 0]

Для каждого из 118 вопросов Coverage B1024 у v1 и структурного v2 совпала точно;
то же верно для Coverage@5 и всех трёх retriever. Разности не компенсировали друг
друга при усреднении: все парные разности равны нулю. Поэтому повторная выборка
этих разностей даёт [0; 0]. Это характеристика данной ранее просмотренной
AI-only выборки; перенос результата на другие вопросы не установлен.

Упаковка всё же изменилась в **{len(changed)}/1144 документах**. Gold-документы
**{len(changed_queries)}/118 вопросов** входят в это множество. Ранги и фактические
контексты также менялись, поэтому равенство Coverage не означает одинаковую выдачу.

Число вопросов с изменением показателя (v2 headings против v1):

| Retriever | Coverage B1024 | Coverage@5 | EvidenceMRR@5 | DocumentMRR@10 | Токены контекста |
|---|---:|---:|---:|---:|---:|
{table}

Оба v2 дали одинаковые координаты, chunk_id и итоговые метрики во всём корпусе.
Узкая эвристика вводной предложила 6 связей: 5 уже сохранены структурным v2,
ещё одна требует 532 токена при encoder budget 510. Дополнительного вмешательства
в границы у этого компонента не возникло. Правило после результата не расширялось.

## Связка есть в индексе, но контекст после поиска неполон

Единственный случай hybrid B1024 с целиком возвращённым B и отсутствующим A
в gold-документе — `{example['query_id']}`, группа `{example['group']['group_id']}`.
Он одинаков у всех трёх вариантов. A: `{co['a_quotes'][0]}`.

Индекс содержит целую связку в чанке [{co['span'][0]}, {co['span'][1]}) длиной
**{co['tokens']} токенов**, hybrid rank **{co['hybrid_rank']}**. К этому рангу
после выбранных rank 1 и 2 остаётся **{co['budget_remaining_at_rank']} токенов**;
целый чанк не помещается и пропускается прежней budget packing.
Второй выбранный чанк начинается с B без заголовка A. Итоговый контекст занимает
**{example['selected_context']['used_tokens']} токенов**.

Это конкретно показывает различие index preservation и retrieved completeness.
Заголовок здесь оценивается по структурному определению; его необходимость для
ответа не получала новой смысловой метки. Retriever и budget policy сохранены.

## Что осталось вне узкого правила

Из семи прежних content-checked AI-групп 6 сохраняются, одна остаётся разорвана
во всех вариантах: `{unhandled['group_id']}`. B: `{unhandled['b_quote']}`;
A: `{unhandled['a_quotes'][0]}`. Минимальный непрерывный интервал занимает
**{unhandled['continuous_interval_tokens']} токенов**, то есть помещается в 480.
Это удалённая область действия заголовка для названия tutorial; правило коротких
полей соединения её не распознаёт, а структурное правило связывает только первый
блок поддерева. Не требуется новая метка, чтобы показать это ограничение: источник
решения остаётся прежней **AI-only**, а не человеческой проверкой.

Все 6 размеченных смысловых документов отсутствуют среди gold-документов 118 QA.
В hybrid B1024 размеченные B полностью не возвращены, поэтому условная полнота
смыслового контекста имеет нулевой знаменатель и не оценивается. Человеческих
проверенных групп — 0. Данные не подтверждают улучшение необходимого смыслового
контекста; экспериментальная эвристика остаётся ограниченной предложенной формой.

Структурная часть завершена: исправлены 132/132 известных разрывов, новых аналогичных
разрывов нет; 4339/4339 heading→body связей сохранены. На этой рабочей точке это
улучшение поведения индекса без изменения основного измеренного покрытия retrieval.

Полные координаты, выбранные чанки, ранги, исходные AI-цитаты и все подсчёты —
в diagnostics.json. Методика не перенастраивалась; исходный experiment manifest
проверен по всем хешам до и после этой интерпретации.
"""
    (output / "report.md").write_text(text, encoding="utf-8")
    check_hashes(immutable)
    check_hashes(preflight["source_hashes"])
    check_hashes(preflight["immutable_source_artifacts"])
    write_json(output / "manifest.json", {
        "status": diagnostics["status"], "created_at_utc": now(),
        "source_run_manifest_sha256": digest(run / "manifest.json"),
        "script_sha256": digest(Path(__file__)), "no_retrieval_or_encoder_inference": True,
        "source_artifacts_unchanged": True, "new_semantic_labels": 0,
        "artifact_sha256": {p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()},
    })
    print(json.dumps({"output": str(output), "paired_coverage_changes": 0, "source_run_unchanged": True}))


if __name__ == "__main__":
    main()
