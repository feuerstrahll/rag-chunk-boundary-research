"""Render the measured V2 comparison without changing the frozen algorithm."""
from __future__ import annotations

import json

NAMES = {"v1": "Hierarchy v1", "headings": "V2: заголовки", "headings_intro": "V2: заголовки + вводная"}


def percent(value):
    return "— (B не найден)" if value is None else f"{100 * value:.2f}%"


def render_report(output, summary, heading_rows, groups, docs, local_chunks):
    aggregates, costs = summary["aggregates"], summary["costs"]
    lookup = {(r["variant"], r["method"]): r for r in aggregates}
    contrasts = {r["contrast"]: r for r in summary["paired_intervals"]
                 if r["method"] == "hybrid" and r["metric"] == "EvidenceCoverage@B=1024"}
    structure = contrasts["headings_minus_v1"]
    intro = contrasts["headings_intro_minus_headings"]
    headings = summary["heading_behavior"]
    detector = summary["detector"]["headings_intro"]
    label_status = summary["semantic_label_status"]
    lines = [
        "# V2: сохранение конкретных контекстных связок", "",
        ("Новые измерения исследовательского продолжения от 27.10.2026. "
        f"{summary['documents']} документов, {summary['questions']} ранее просмотренных AI-only вопросов, "
        f"{summary['cluster_count']} документных кластеров. V1 и его результаты сохранены без изменений. "
        "Старые значения hierarchy переиспользованы после проверки совместимости протокола, кода, "
        "входов, модели и версий пакетов. Оба V2 заново сегментированы, проиндексированы и оценены."), "",
        "## Основной результат", "",
        ("Основной заранее выбранный показатель — **hybrid EvidenceCoverage при B=1024**. "
        f"V1: {lookup['v1', 'hybrid']['EvidenceCoverage@B=1024']:.6f}; "
        f"V2 заголовки: {lookup['headings', 'hybrid']['EvidenceCoverage@B=1024']:.6f}; "
        f"V2 заголовки+вводная: {lookup['headings_intro', 'hybrid']['EvidenceCoverage@B=1024']:.6f}."), "",
        (f"Вклад цепочек: {structure['difference'] * 100:+.3f} п.п., описательный 95% кластерный интервал "
        f"[{structure['ci_lower'] * 100:+.3f}; {structure['ci_upper'] * 100:+.3f}] п.п. "
        f"Дополнительный вклад вводной: {intro['difference'] * 100:+.3f} п.п., "
        f"[{intro['ci_lower'] * 100:+.3f}; {intro['ci_upper'] * 100:+.3f}] п.п."), "",
        ("Интервал структурного компонента включает ноль: направленный выигрыш retrieval этим сравнением не установлен."
         if structure["ci_lower"] <= 0 <= structure["ci_upper"] else
         ("В этой выборке структурный компонент увеличил основное покрытие. "
          if structure["ci_lower"] > 0 else "В этой выборке структурный компонент снизил основное покрытие. ") +
         "Интервал не включает ноль; это не независимое подтверждение и не устраняет ограничений AI-разметки."), "",
        "## Правила и отличие от v1", "",
        ("V1 сначала делил документ по непосредственному родителю, поэтому родительский заголовок "
        "мог остаться отдельно от первого текста дочернего раздела. V2 сначала накапливает ещё не "
        "получившие текста заголовки, связывает их с первым содержательным блоком своего поддерева "
        "и упаковывает весь непрерывный интервал. После первого содержательного блока действует "
        "обычная упаковка с родительскими ограничениями. Пустые поддеревья истекают на следующем "
        "заголовке того же/высшего уровня; пустой корень не получает текст другого корня."), "",
        ("Комментарии игнорируются только при распознавании содержательного блока. Они остаются "
        "в canonical text, чанке и токенном бюджете. Target=480, encoder text budget=510, overlap=64. "
        "Одна предпочтительная единица может занимать 481–510 токенов. При невозможном бюджете "
        "сохраняется самый ранний вписывающийся дочерний суффикс; остальной интервал режется прежним "
        "token fallback. Все нарушения регистрируются, токены не удаляются."), "",
        ("Отключаемая эвристика `connection_fields_1` проверяет только короткие имена полей "
        "соединения. A должен явно называть операцию, необходимость ввода и объект либо однозначно "
        "получать именованный объект из ближайшего заголовка операции. B должен целиком совпасть "
        "с закрытым перечнем вроде Port number, Database name, Instance ID. Область действия — "
        "непрерывный ряд list после A, прерываемый заголовком или другим содержательным блоком. "
        "Одиночные following/this setting/in this case, самодостаточная инструкция, ссылка и неоднозначный "
        "объект не проходят правило. Направление только B → A; обратной зависимости на весь список нет."), "",
        ("Полные признаки, отказы, конфликтные интервалы и fallback описаны в "
        "`docs/context_packing_v2_protocol.md`; параметры и хеши зафиксированы в frozen_config/preflight "
        "до retrieval. Правило построено на исходных train-документах и не менялось по 118 вопросам. "
        "Парсер, MiniLM, BM25/dense/RRF и budget packing прежние; одинаковые spans сохраняют прежние "
        "chunk_id, чтобы не менять tie-breaking из-за переименования."), "",
        "## Качество поиска", "",
        "| Сегментация | Поиск | Coverage B1024 | Coverage@5 | EvidenceRecall@5 τ=.8 | EvidenceMRR@5, один чанк | DocumentRecall@5 | DocumentMRR@10 | Токены контекста, среднее |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregates:
        names = ["EvidenceCoverage@B=1024", "EvidenceCoverage@5", "EvidenceRecall@5", "EvidenceMRR@5",
                 "DocumentRecall@5", "DocumentMRR@10"]
        lines.append(f"| {NAMES[row['variant']]} | {row['method']} | " +
                     " | ".join(f"{row[name]:.6f}" for name in names) + f" | {row['mean_actual_context_tokens']:.2f} |")
    lines += ["", ("EvidenceCoverage — длина объединения покрытых gold-символов / длина всех gold spans, "
              "с невзвешенным средним по вопросам. EvidenceRecall@5 — индикатор покрытия ≥0.8 "
              "объединением top-5. EvidenceMRR@5 требует одного чанка с покрытием ≥0.8. "
              "Document metrics вычисляются после max-pool по документам до top-k. "
              "Контекст B1024 набирается целыми чанками по рангу с пропуском непоместившихся, "
              "без усечения; повторный overlap оплачивается повторно."), "",
              ("Парные интервалы всех метрик и трёх контрастов для всех retriever — в paired_intervals.csv. "
              "Bootstrap: 10000 повторов, seed=2026, кластеры общих gold-документов. "
              "Интервалы описательные, без поправки на множественность."), "",
              "## Чанки, overlap и затраты", "",
              "| Сегментация | Чанки | Индексируемые токены | Min / P10 / median / P90 / max | Overlap, токены | Fallback чанки | Сегментация, с | BM25, с | Dense + проверка, с | Запросы + запись, с | Peak RSS, MiB | Dense, MiB |",
              "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for variant, row in costs.items():
        lengths, timing = row["chunk_tokens"], row["stages_seconds"]
        lines.append(f"| {NAMES[variant]} | {row['chunk_count']} | {row['indexed_tokens']} | " +
                     " / ".join(f"{lengths[name]:.0f}" for name in ["min", "p10", "median", "p90", "max"]) +
                     f" | {row['actual_overlap_tokens_sum']} | {row['fallback_chunks']} | "
                     f"{timing['segmentation_seconds']:.2f} | {timing['bm25_build_seconds']:.2f} | "
                     f"{timing['dense_build_including_encoder_validation_seconds']:.2f} | "
                     f"{timing['query_ranking_scoring_and_artifact_io_seconds']:.2f} | "
                     f"{row['peak_process_rss_bytes'] / 2**20:.1f} | {row['dense_matrix_bytes'] / 2**20:.2f} |")
    lines += ["", ("Для V1 затраты взяты из завершённого старого измерения. Оба V2 запускались в отдельных "
              "процессах, CPU threads=4, batch=32, та же локальная модель и среда. Время измерено один раз; "
              "кэши ОС не очищались. RSS включает весь процесс с моделью и индексом, шаг опроса 100 мс. "
              "Dense bytes — только float32 матрица, не полный индекс. В retrieval-конвейере нет "
              "внешних LLM/API-вызовов или денежных API-затрат; новых моделей и генерации контекста нет. "
              "Для каждого варианта из неизменённого кода следует 119 вызовов локального model.encode "
              "(один для корпуса, 118 для запросов); число forward batches и доля фактического overlap "
              "от индексируемых токенов дополнительно сохранены в summary.costs как derived."), "",
              "## Сохранность заголовков в индексе", "",
              ("Весь корпус проверен независимо от старого списка diagnostics. Единица таблицы — "
              "каждый конкретный заголовок и первый содержательный блок его поддерева. "
              "Отдельные цепочки со всеми ожидающими заголовками также оцениваются ниже."), "",
              "| Сегментация | Все heading→body связи | Из известных 132 сохранено | Все сохранено | Разорвано при ≤480 | Разорвано при ≤510 | Разорвано при >510 | Новые разрывы относительно v1 (в т.ч. выполнимые) |",
              "|---|---:|---:|---:|---:|---:|---:|---|"]
    for variant, row in headings.items():
        lines.append(f"| {NAMES[variant]} | {row['all_heading_body_links']} | {row['known_cases_preserved']} | "
                     f"{row['all_preserved']} | {row['broken_fits_target']} | {row['broken_fits_encoder']} | "
                     f"{row['broken_exceeds_encoder']} | {row['new_broken_relative_to_v1']} "
                     f"(выполнимые: {row['new_feasible_broken_relative_to_v1']}) |")
    lines += ["", ("Исправление 132 случаев — проверка поведения алгоритма. Оно не доказывает "
              "рост поискового покрытия. Пустые разделы не включены в знаменатель связей с существующим B. "
              "Полный список с координатами, токенами и результатом каждого варианта: heading_behavior.jsonl."), "",
              "## Индекс и возвращённый контекст оцениваются отдельно", "",
              ("В индексе требуется **один** чанк с B и всеми A. После поиска разрешено объединение "
              "нескольких выбранных чанков. У цепочки A — все ожидавшие заголовки. "
              "У смысловой группы A — только явно размеченные необходимые spans, включая совместные "
              "A у pilot_025. Соседние пункты перечня не объявляются необходимыми."), "",
              "| Сегментация | Происхождение группы | Группы | Сохранено одним чанком | Разорвано при выполнимом encoder | Разорвано при превышении encoder |",
              "|---|---|---:|---:|---:|---:|"]
    for row in summary["index_preservation"]:
        lines.append(f"| {NAMES[row['variant']]} | {row['cohort']} | {row['groups']} | {row['preserved']} | "
                     f"{row['broken_fits_encoder']} | {row['broken_exceeds_encoder']} |")
    lines += ["", "### Полнота после hybrid поиска", "",
              ("`Все пары` — все query×group выбранной области, включая случаи, где B не найден. "
              "Условная полнота = complete / B полностью найден; при нуле знаменателя показано «—». "
              "Структурные группы в gold-документе — операционная проверка полноты этого контекста, "
              "не дополнительная разметка необходимости заголовков для ответа."), "",
              "| Вариант | Контекст | Группы | Область | Все пары | B полностью найден | Найдены B + все A | B найден без всех A | Полнота при найденном B |",
              "|---|---|---|---|---:|---:|---:|---:|---|"]
    for row in summary["retrieved_completeness"]:
        if row["method"] != "hybrid":
            continue
        if row["cohort"] == "structural_chain" and row["scope"] != "gold_document_only":
            continue
        if row["cohort"] != "structural_chain" and (row["scope"] != "all_indexed_documents" or row["context"] != "B1024"):
            continue
        lines.append(f"| {NAMES[row['variant']]} | {row['context']} | {row['cohort']} | {row['scope']} | "
                     f"{row['eligible_query_group_pairs']} | {row['b_fully_returned']} | {row['complete']} | "
                     f"{row['b_returned_missing_a']} | {percent(row['complete_given_b_returned'])} |")
    lines += ["", (f"Размеченные смысловые документы: {label_status['labeled_documents']}; "
              f"вопросов основной выборки в этих документах: {label_status['evaluation_questions_in_labeled_documents']}. "
              "Статистика по всем документам также учитывает размеченный B, попавший в выдачу как "
              "контекст другого вопроса. Это не специально созданный тест понимания этих B. "
              "Все retriever/области/B1024/top5 и знаменатели сохранены в retrieved_completeness.csv; "
              "retrieved_group_details.jsonl содержит строки групп из документов, присутствовавших "
              "в выбранном контексте. Остальные пары не возвращены и учтены в полном знаменателе."), "",
              ("Семь прежних групп — содержательно проверенные **AI**, не человеческие метки; два новых "
              "примера — предложения AI_Codex на документах разработки. Ни один автоматически "
              "предложенный детектором случай по всему корпусу не добавлен к gold. Человеческий "
              "знаменатель равен **0**, поэтому человечески подтверждённое улучшение необходимого "
              "смыслового контекста не оценивается."), "",
              "## Срабатывания отключаемого правила", "",
              (f"Предложено связей B→A: **{detector['intro_links_proposed']}**, "
              f"документов: **{detector['documents_with_intro_links']}**; сохранено в одном чанке: "
              f"**{detector['intro_links_preserved']}**; непрерывный интервал превышает encoder: "
              f"**{detector['intro_links_exceed_encoder']}**."), "",
              "| Решение упаковки | Число |", "|---|---:|"]
    lines += [f"| {name} | {count} |" for name, count in detector["packing_decisions"].items()]
    lines += ["", "| Причина воздержания | Число |", "|---|---:|"]
    lines += [f"| {name} | {count} |" for name, count in detector["abstention_reasons"].items()]
    lines += ["", ("Единица отказа — один кандидат A без следующего списка либо одна пара A/B "
              "при наличии списка; это разные технические единицы, общий процент зависимости по ним не считается. "
              "Числа выше — статистика детектора, не precision/recall или подтверждённая частота "
              "необходимых зависимостей. У длинного перечня каждый B рассматривается отдельно: "
              "невместимость позднего пункта не отменяет короткие допустимые интервалы. "
              "Полные предложения и отказы: headings_intro/preferred_links.jsonl и intro_candidates.jsonl."), "",
              "## Проверяемые примеры", "",
              ("Примеры выбраны по структурным категориям и заранее известным train-записям, "
              "без отбора по выигрышу retrieval. Цитаты — из canonical text."), ""]
    chosen, selected_docs = [], set()
    for row in heading_rows:
        if row["known_132"] and row["doc_id"] not in selected_docs:
            chosen.append(row)
            selected_docs.add(row["doc_id"])
            if len(chosen) == 2:
                break
    for number, row in enumerate(chosen, 1):
        text = docs[row["doc_id"]].canonical_text
        a, b = row["a_spans"][0], row["b_span"]
        cochunk = next(c for c in local_chunks["headings"][row["doc_id"]]
                       if c["char_start"] <= row["start"] and c["char_end"] >= row["end"])
        lines += [f"### Структурный положительный пример {number}", "",
                  (f"Документ `{row['doc_id']}`. A [{a[0]}, {a[1]}): `{text[a[0]:a[1]]}`. "
                  f"Первый B [{b[0]}, {b[1]}):"), "", "> " + text[b[0]:b[1]][:500].replace("\n", " "), "",
                  (f"Непрерывная связка: {row['tokens']} токенов. V1 сохраняет: {row['v1']}; "
                  f"V2: {row['headings']}. Новый чанк [{cochunk['char_start']}, {cochunk['char_end']}) "
                  f"занимает {cochunk['token_count']} токенов."), ""]
    for empty in summary.get("known_empty_root_examples", []):
        heading = empty["heading_run"][0]
        lines += ["### Структурный отрицательный пример: пустой корень", "",
                  (f"Документ `{empty['doc_id']}`, A `{heading['text']}` "
                  f"[{heading['char_start']}, {heading['char_end']}). До следующего корня "
                  f"`{empty['next_heading']}` своего содержательного блока нет. "
                  "Связь с чужим B не создаётся; это пустое поддерево, а не потерянная выполнимая связка."), ""]
    oversized = next((r for r in heading_rows if not r["fits_encoder"]), None)
    if oversized:
        text = docs[oversized["doc_id"]].canonical_text
        a, b = oversized["a_spans"][0], oversized["b_span"]
        lines += ["### Отказ от безусловного сохранения: бюджет", "",
                  (f"Документ `{oversized['doc_id']}`, заголовок `{text[a[0]:a[1]]}`, "
                  f"B [{b[0]}, {b[1]}). Связка занимает **{oversized['tokens']}** токенов при encoder=510. "
                  f"V2 сохраняет целиком: {oversized['headings']}. Применено допустимое fallback-разбиение; "
                  "невыполнимая предпочтительная связка не выдаётся за сохранённую."), ""]
    else:
        lines += ["### Отказ по бюджету: отдельная проверка", "",
                  ("В корпусе ни одна определённая структурная связка не превышает 510 токенов. "
                  "Fallback проверен синтетическими единицами 480/481/510/511 токенов: при 511 "
                  "предок с полным B не помещается, его разрыв регистрируется, а выполнимая дочерняя "
                  "связка сохраняется. Это тест поведения, не новый размеченный корпусный случай."), ""]
    proposed = [json.loads(line) for line in
                (output / "headings_intro/preferred_links.jsonl").read_text().splitlines()]
    too_long = next((r for r in proposed if r["kind"] == "intro" and not r["fits_encoder"]), None)
    if too_long:
        text = docs[too_long["doc_id"]].canonical_text
        b = too_long["b_span"]
        lines += ["### Предложение вводной, не помещающееся в бюджет", "",
                  (f"Документ `{too_long['doc_id']}`, B `{text[b[0]:b[1]]}`. "
                  f"Интервал от A до конца B — {too_long['tokens']} токенов, включая все промежуточные "
                  f"поля и комментарии. Решение: `{too_long['packing_decision']}`. "
                  "Это автоматически предложенная эвристическая связь без человеческой метки."), ""]
    development = json.loads((output / "development_examples.json").read_text())
    rows = development["examples"]
    positive = next(r for r in rows if r["record_id"] == "pilot_025")
    anchors = [r["a_quote"] for r in rows if r["record_id"] == "pilot_025"]
    lines += ["### Содержательное правило: известный положительный AI-пример", "",
              (f"`pilot_025`, документ `{positive['doc_id']}`; B: `{positive['b_quote']}`. "
              "Совместные A:"), "", *["- " + a for a in anchors], "",
              ("Без A не названы соединение и операция. Направление Port number → вводная+заголовок. "
              "Остальные поля перечня не являются необходимыми A. Это прежняя content-checked "
              "AI-разметка. Новые Database name и Instance ID имеют те же явно названные A, "
              "но сохранены как отдельные **AI-предложения**, а не проверка людьми."), ""]
    for record_id in ["pilot_004", "pilot_008", "pilot_010"]:
        row = next(r for r in rows if r["record_id"] == record_id)
        lines += [f"### Отрицательный/неприменимый к правилу пример {record_id}", "",
                  "> " + row["b_quote"].replace("\n", " "), ""]
        explanation = {
            "pilot_004": "Самодостаточный шаг называет deployment space, меню и Import space. "
                         "Правило короткого имени поля воздерживается. Исходная метка AI draft; "
                         "здесь это иллюстрация отказа, она не включена в оценочную положительную разметку.",
            "pilot_008": "Сам B называет classification/regression и настройку categorical/numerical imputation; "
                         "соседство с описанием timeseries не создаёт зависимости. Прежняя AI-метка reviewed_by_codex.",
            "pilot_010": "Маркер these methods не создаёт обратной зависимости вводной на все строки таблицы. "
                         "Прежняя content-checked AI-метка negative. Узкое правило не связывает такую вводную "
                         "с последующей таблицей как с её необходимым A.",
        }
        lines += [explanation[record_id], ""]
    lines += ["## Вклад компонентов и ограничения", "",
              (f"Структурный компонент исправил {headings['headings']['known_cases_preserved']}/132 известных "
              f"случаев, новых выполнимых разрывов относительно v1: "
              f"{headings['headings']['new_feasible_broken_relative_to_v1']}. "
              "Это завершённый инженерный результат независимо от направления поисковой метрики."), ""]
    for row in summary["span_equality"]:
        lines += [(f"Для {NAMES[row['reference']]} → {NAMES[row['candidate']]} совпали все координаты "
                  f"в {row['identical_spans']}/{row['documents']} документах."), ""]
        if row["candidate"] == "headings_intro" and row["identical_spans"] == row["documents"]:
            lines += [("У эвристики вводной не возникло дополнительного изменения границ: "
                       "вписывающиеся предложенные связки уже сохранял структурный v2, а длинный "
                       "интервал оставлен с зарегистрированной причиной отказа. Координаты и chunk_id "
                       "обоих v2 совпадают во всём корпусе. Вывод ограничен этим правилом и рабочей точкой; "
                       "из него не следует бесполезность содержательного контекста вообще."), ""]
    lines += [f"Для отключаемой вводной измеренная дополнительная разность основной метрики составляет "
              f"{intro['difference'] * 100:+.3f} п.п. " +
              ("Направленный выигрыш не установлен: описательный интервал включает ноль."
               if intro["ci_lower"] <= 0 <= intro["ci_upper"] else
               "Наблюдаемая разность относится к этому исследовательскому протоколу и AI-only выборке."), "",
              ("Малое число AI-примеров, отсутствие человеческой проверки, узкий лексический охват "
              "и прежнее знакомство с 118 QA ограничивают содержательный вывод. Сохранность связки "
              "в индексе не тождественна её нахождению и не доказывает полноту ответа. Результат "
              "не обосновывает универсальную семантическую сегментацию, перенос на другой корпус "
              "или качество генерации. Параметры и правила после измерений не перенастраивались."), "",
              "## Проверки и воспроизведение", "",
              ("Все 1062 строки (3 варианта × 3 retriever × 118 QA) независимо пересчитаны из полных "
              "сохранённых rankings отдельным кодом объединения spans. Проверены порядок/равные scores, "
              "B1024 packing и сохранённые контексты; literal canonical slices, token_count и покрытие "
              "токенизируемого текста всех чанков. Encoder IDs проверялись прежним DenseIndex до "
              "инференса. Оба выключенных переключателя общей реализации воспроизвели все 1144 "
              "документа v1: совпали spans, chunk_id, text, token_count и fallback flags; повторного "
              "encoder-инференса для этой проверки нет. Хеши старых артефактов и исходников сверены "
              "до и после; они не изменены."), "",
              "Запуск из корня рабочей области (указать **новый** каталог результата):", "",
              "```bash", "PYTHONDONTWRITEBYTECODE=1 article/.venv-hierarchy/bin/python article/scripts/run_context_packing.py \\",
              "  --output-dir article/results/runs/<new-run>",
              "PYTHONDONTWRITEBYTECODE=1 article/.venv-hierarchy/bin/python article/scripts/analyze_context_packing.py \\",
              "  --run article/results/runs/<new-run>", "```", "",
              ("Методика оценивания зафиксирована до retrieval; исходники отчёта и независимого проверяющего "
              "кода завершены после начала запуска, их отдельные хеши сохранены в manifest. "
              "Это раскрытое происхождение анализа, без изменения параметров алгоритма."), "",
              ("Артефакты: frozen_config.json/preflight.json — протокол и входы; development_examples.json — "
              "примеры и происхождение меток; evaluation_groups.jsonl — точные A/B; index_preservation.csv — "
              "сохранность; retrieved_completeness.csv — полнота после поиска; aggregate_metrics.csv и "
              "paired_intervals.csv — качество; summary.json — все числа; validation.json и manifest.json — "
              "проверки и SHA-256. В каталогах вариантов — chunks, полные rankings, контексты, "
              "детекторные решения и index_statistics. В retrieval-эксперименте обращений к внешним моделям нет.")]
    checks = summary.get("software_checks")
    if checks:
        suite = checks["full_suite"]
        lines += ["", "## Состояние программных проверок", "",
                  (f"Полный набор article/tests: **{suite['passed']} passed, {suite['failed']} failed**. "
                  "Все новые тесты и регрессионные тесты hierarchy v1 прошли. Пять ошибок относятся "
                  "к прежнему test_inspect_materials: FileNotFoundError для отсутствующего executable git. "
                  "Этот старый модуль не изменялся. Проверка новых исходников Ruff пройдена. "
                  "Логи и test_verification.json сохранены рядом."), "",
                  ("Git HEAD/dirty status текущей среды недоступны; происхождение исполнявшегося кода "
                  "определено SHA-256 и проверенным upstream snapshot. Поле git_metadata в preflight "
                  "взято из reference v1, а не из нового Git-вызова. Это не утверждение о чистом дереве.")]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
