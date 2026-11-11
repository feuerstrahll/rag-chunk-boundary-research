# Достаточность контекста: подготовка выполнена, LLM недоступна

27.10.2026. Подготовлены **386 контекстов** из сохранённого hybrid retrieval:
по два для 118 AI-only и 75 исходных gold QA. **Оценок LLM — 0.**
В среде нет настроенного inference endpoint/API-ключа; локальные сервисы на
портах 11434, 1234, 8000 и 8080 не принимают соединения. Доступная ChatGPT-сессия
Codex не является настроенным независимым API-судьёй с temperature=0.
Оценки не имитировались.

| Выборка | Метод | Вопросов | Оценено | Sufficiency Rate | EvidenceCoverage@1024 |
|---|---|---:|---:|---:|---:|
| AI-only | fixed 480/20 | 118 | 0 | н/д | 0,648143 |
| AI-only | structure 510/64 | 118 | 0 | н/д | 0,694211 |
| Исходные gold | fixed 241/10, исторический | 75 | 0 | н/д | н/д |
| Исходные gold | structure 510/64, исторический | 75 | 0 | н/д | н/д |

**Парная разность Sufficiency Rate structure−fixed и 95% CI: н/д для обеих
выборок.** Нельзя интерпретировать отсутствие оценок как нулевую достаточность
или отсутствие различий. Coverage в первых двух строках взята из существующих
per-question результатов и проверена на тех же контекстах. Для исторических
gold-контекстов такой сохранённой метрики нет; Coverage@5 не подставлялась.

AI-only и исходные gold не объединяются. На исходных QA сохранён другой fixed
operating point; 45 train и 30 ранее просмотренных test учитываются в mapping.
Все данные ранее использовались в исследовании; независимого подтверждающего
теста этот анализ не создаёт. Человеческая верификация AI-only gold не заявляется.

## Конкретные наблюдения в контекстах

Это примеры для последующей проверки судьи, **не LLM-оценки достаточности**.
Полные тексты и отображение request ID находятся в `requests.jsonl` и `mapping.jsonl`.

| Вопрос AI-only | Coverage fixed / structure | Проверяемое наблюдение |
|---|---:|---|
| `holdout_auto_001`: What does the SPSS Modeler function list display? | 1 / 0 | В fixed есть «The function list displays all available SPSS Modeler functions and operators». В structure извлечены сведения о scripting context, списках и справочнике функций; указанного предложения нет. |
| `holdout_auto_005`: What model asset can the XGBoost Tree node generate? | 0 / 1 | В structure есть «You can use the XGBoost Tree node to generate an XGBoost Tree model nugget». В fixed есть описание алгоритма и соседний XGBoost Linear model nugget; их нельзя автоматически отождествлять. |
| `holdout_auto_012`: How do you add a Dropbox connection in a deployment space? | 1 / 0 | **В обоих** контекстах фактически есть «In a deployment space: Click Add to space > Connection». У structure нулевое покрытие конкретного размеченного span его gold-документа не означает отсутствия этой инструкции в извлечённом тексте. |

Последний случай показывает потенциально полезное различие конструкций:
EvidenceCoverage считает координаты размеченного evidence, тогда как судья
должен оценить смысл всего переданного контекста. Это наблюдение об извлечённом
тексте; оно ещё не подтверждает надёжность судьи или величину добавочной информации.
Случаи «coverage>0, но судья признал контекст недостаточным» пока установить
невозможно — судья не запускался.

## Проверка и воспроизведение

Для всех 386 контекстов проверены буквальные срезы канонического текста,
сохранённые token counts и фактический размер собранного текста ≤1024.
Использован прежний MiniLM WordPiece tokenizer: бюджет сравним с существующей
EvidenceCoverage, но не выражен в нативных токенах будущего судьи.
На 118 AI-only вопросах каждого метода воспроизведён только отбор из сохранённого
ranking: список чанков, бюджет и coverage совпали с сохранёнными результатами.
Chunking, retrieval и индексы не запускались.

Протокол: [context_sufficiency_protocol.md](../../../docs/context_sufficiency_protocol.md).
Параметры, версии среды, SHA-256 входов и кода: `frozen_config.json`, `manifest.json`,
`requirements.txt`. Таблицы: `comparison.csv`, `paired_comparison.csv`.
В `judgments.csv` только заголовок; пустые поля означают отсутствие наблюдений.
`v1` — предыдущий технический пакет подготовки без оценок; актуальный пакет — `v2`.

Из корня рабочей области, с зависимостями из `requirements.txt`:

```bash
PYTHONPATH=/tmp/context-sufficiency-deps PYTHONDONTWRITEBYTECODE=1 python article/scripts/evaluate_context_sufficiency.py prepare --output article/results/runs/context_sufficiency_NEW
PYTHONPATH=/tmp/context-sufficiency-deps PYTHONDONTWRITEBYTECODE=1 python -m pytest article/tests/test_context_sufficiency.py -q -p no:cacheprovider
PYTHONPATH=/tmp/context-sufficiency-deps python article/results/runs/context_sufficiency_20261027_v2/verify_prepared.py
```

Первый запуск требует локального tokenizer по пути прежней конфигурации;
загрузок модели/данных не делает. Все выходные каталоги должны быть новыми.
Когда будет указан существующий OpenAI-compatible endpoint, запуск имеет вид:

```bash
python article/scripts/evaluate_context_sufficiency.py judge --prepared article/results/runs/context_sufficiency_20261027_v2 --output article/results/runs/context_sufficiency_JUDGED_NEW --endpoint "$EXISTING_LLM_CHAT_COMPLETIONS_URL" --model "$EXISTING_LLM_MODEL" --key-env EXISTING_LLM_API_KEY
```

Судья получает только вопрос и контекст; параметры temperature=0 и JSON-формат
передаются явно. Скрипт сохраняет сырые ответы, проверяет метки, считает Sufficiency
Rate и парный 95% кластерный bootstrap CI (10 000 повторений, seed=2026), сохраняет
расхождения с coverage и пакет 12 оценок для содержательной проверки. Если сервис
не принимает эти параметры, запуск останавливается без ослабления протокола.
Сетевой путь пока не проверен на реальном endpoint.

**Ручная проверка 10–15 оценок не выполнена: самих оценок нет.** Прочитанные выше
примеры не заменяют такую проверку. До интерпретации будущих оценок нужно проверить
12 выбранных случаев по полным контекстам и сохранить отдельный журнал; проверка
Codex должна быть обозначена как AI content review, а не человеческая верификация.

**Вывод:** измерить различие Sufficiency Rate между fixed и structure-aware и
оценить согласованность с EvidenceCoverage пока нельзя. Есть конкретный кандидат
на добавочную информацию — повторяющаяся инструкция Dropbox при coverage=0, —
но экспериментальный вывод требует реальных оценок независимого судьи и их проверки.
