# JSON-разметка пилота

Редактируйте `annotation_workbook.json` как обычный UTF-8 JSON. В `records` находятся 54 случая с B и ближайшими блоками. Полный текст, все блоки, canonical offsets и исходный Markdown доступны в `documents[doc_id]`. Поля `answer` оставлены пустыми; AI-черновик 12 случаев находится **отдельно** и не включён в этот файл. Технических предсказаний детектора здесь нет.

## Порядок

1. Прочитайте `b.quote` и заполните `answer.missing_context` до просмотра `nearby_blocks`.
2. Найдите минимально необходимые A в `documents[doc_id].blocks`. Для каждого внесите в `answer.necessary_a` объект `{"block_id":"...","span":[start,end]}`. У нескольких совместно необходимых A заполните `dependency_group`.
   При `uncertain` допускаются частично установленные кандидаты A; экспорт сохраняет их как `partial_candidate`, не считает положительной зависимостью.
3. Заполните `decision` (`positive`, `negative`, `uncertain`, `not_applicable`), `relation_type`, `rationale`, `annotator_id`. `negative` означает самодостаточность B, а не бесполезность соседнего A. Ошибку парсинга опишите в `parser_issue`; исходный Markdown остаётся доступен для сверки.
4. Оставьте `review_status` равным `draft`, пока независимая проверка и разбор расхождений не выполнены людьми.

Для первых 12 записей можно экспортировать частичный черновик; для полного набора уберите `--allow-incomplete`:

```powershell
& ..\techdoc-chunking-pipeline\.venv\Scripts\python.exe -m scripts.dependency_json_workbook export --workbook results/pilots/dependency_annotation_json_20261022_v2/annotation_workbook.json --output results/pilots/dependency_annotation_json_20261022_v2/human_draft.jsonl --allow-incomplete
```

Команда откажется экспортировать пустой файл, неверные решения, отсутствующий A у `positive` или невалидные координаты. Выход не перезаписывается. Воспроизвести пустой шаблон можно в **новом** пути командой:

```powershell
& ..\techdoc-chunking-pipeline\.venv\Scripts\python.exe -m scripts.dependency_json_workbook create --output results/pilots/dependency_annotation_json_20261022_v2/rebuilt_workbook.json
```
