# Дополнительный анализ после основного эксперимента

Статус: post-outcome exploratory. Алгоритм, RRF и основной запуск не изменялись. Нового индексирования, поиска или запуска encoder нет; использованы сохранённые результаты и pinned tokenizer.

## Терминология

EvidenceSuccess@5, τ=0,8 — доля вопросов, у которых объединение top-5 покрывает не менее 80% gold. В исходном машинном артефакте этот показатель называется EvidenceRecall@5; значения не изменены. EvidenceMRR@5, τ=0,8, один чанк — средний обратный ранг первого чанка в top-5, который один покрывает не менее 80% всего gold вопроса; при отсутствии такого чанка — 0. Документные Recall@5 и MRR@10 остаются отдельными метриками.

## 133 диагностические записи о заголовках

Записи относятся к 23 документам и 133 различным чанкам. Это не число доказанных retrieval-ошибок. Для аудита содержимым считается непустой не-heading блок после удаления HTML-комментариев только в диагностической проверке; текст pipeline не изменён.

| Проверка | Число записей |
|---|---:|
| Чанк содержит только заголовки | 121 |
| В чанке есть другой текст после исключения комментариев | 11 |
| В чанке заголовок и только комментарии вместо другого текста | 1 |
| Следующий заголовок — более глубокого уровня | 132 |
| В поддереве найден последующий текст | 132 |
| В поддереве нет текста после исключения комментариев | 1 |
| Первый текст поддерева отделён от завершающего заголовка | 132 |
| Связка заголовков до конца первого текста помещается в 480 | 132 |
| Та же связка помещается в 510 | 132 |
| Для всей связки дополнительно мешает предел 510 | 0 |
| Связка до первого токена текста помещается в 510 | 132 |
| Заголовки плюс весь следующий сохранённый чанк помещаются в 510 | 133 |
| Два сохранённых чанка целиком помещаются в 510 | 128 |
| Абляция сохраняет заголовок с полным первым текстовым блоком | 132 |

Во всех случаях после завершающего заголовка до следующего заголовка нет прямого текстового блока. Поэтому отделённое содержимое следует проверять как содержимое подраздела/поддерева, а не автоматически считать потерянным вводным абзацем. В 132 случаях с текстом поддерева связка до конца первого текстового блока помещается даже в целевые 480 токенов; размер не объясняет это отделение. Во всех 132 случаях в абляции заголовок и полный первый текстовый блок находятся вместе. Один случай — пустой корневой раздел перед другим H1; требовать связывания с чужим разделом нельзя. Смена непосредственного родителя действительно завершает область упаковки; таблица отдельно показывает, когда размер также препятствует объединению. Недостаток правила: оно может оставить родительский заголовок вне чанка дочернего содержимого даже при достаточном токенном бюджете. Только один оценочный вопрос относится к этим 23 документам; это не исключает влияния документов как нерелевантных кандидатов и не устанавливает причинный эффект. Не установлено, что каждая такая ситуация ухудшает извлечение evidence.

### Примеры из сохранённых чанков

**Только заголовок; связка с первым текстом помещается. Managing Data Refinery flows**

Документ `0999F59BB8E2E2AB7722D57CDBC051A0984ABE45`; H2 «Working with the Data Refinery flow ##» → H3 «Save a Data Refinery flow ###». Чанк `chk_5ec2380b15dca0d1e7e4c548ccaf6b85`, span [3384, 3422), 8 токенов. Связка до конца первого текстового блока: 109; перенос в следующий сохранённый чанк: 459 токенов.

Завершение текущего сохранённого чанка (не более 800 символов):

> Working with the Data Refinery flow ##

Начало следующего чанка `chk_367b0f7a56b14a4b4f4e78434978935c`, span [3424, 5009):

> Save a Data Refinery flow ###
> 
> Save a Data Refinery flow by clicking the Save Data Refinery flow icon ![Save icon](https://dataplatform.cloud.ibm.com/docs/content/wsj/refinery/images/save.png) in the Data Refinery toolbar\. Data Refinery flows are saved to the project that you're working in\. Save a Data Refinery flow so that you can continue refining a data set later\.
> 
> The default output of the Data Refinery flow is saved as a data asset *source\-file\-name*\_shaped\.csv\. For example, if the source file is `mydata.csv`, the default name and output for the Data Refinery flow is `mydata_csv_shaped`\. You can edit the name and add an extensio

**Заголовок в чанке с другим текстом. IBM Db2 for i connection**

Документ `2A95F85CD3F7250FCBA52D8FDD82D1502FC617D6`; H2 «Prerequisites ##» → H3 «Obtain the certificate file ###». Чанк `chk_8b2d0d4babcee5e3d3134a5d7831ead0`, span [268, 329), 21 токенов. Связка до конца первого текстового блока: 212; перенос в следующий сохранённый чанк: 479 токенов.

Завершение текущего сохранённого чанка (не более 800 символов):

> Supported versions ##
> 
> IBM DB2 for i 7\.2\+
> 
> Prerequisites ##

Начало следующего чанка `chk_72a3af4f9872ed78b6cf719a8dd0bd63`, span [331, 1871):

> Obtain the certificate file ###
> 
> A certificate file on the Db2 for i server is required to use this connection\. To obtain an IBM Db2 Connect Unlimited Edition license certificate file, go to [IBM Db2 Connect: Pricing](https://www.ibm.com/products/db2-connect/pricing) and [Installing the IBM Data Server Driver for JDBC and SQLJ](https://www.ibm.com/docs/SSEPGG_11.5.0/com.ibm.db2.luw.apdv.gs.doc/doc/t0010264.html)\. For installation instructions, see [Activating the license certificate file for Db2 Connect Unlimited Edition](https://www.ibm.com/docs/SSEPGG_11.5.0/com.ibm.db2.luw.licensing.doc/doc/t0057375.html)\.
> 
> Run the bind command ###
> 
> Run

**Пустое поддерево после исключения комментариев. Searching for assets and artifacts across the platform**

Документ `977C81385F7825613F1EDBD3C0DBF44C259BA8D7`; H1 «Searching for assets and artifacts across the platform #» → H1 «Searching for assets across the platform #». Чанк `chk_590ffa57c1e3c0b14461432727573878`, span [0, 56), 9 токенов. Связка до конца первого текстового блока: нет собственного текста поддерева; перенос в следующий сохранённый чанк: 223 токенов.

Завершение текущего сохранённого чанка (не более 800 символов):

> Searching for assets and artifacts across the platform #

Начало следующего чанка `chk_99348684a25111ed81c6c4e542ddc11e`, span [58, 701):

> Searching for assets across the platform #
> 
> You can use the global search bar to search for assets across all the projects and deployment spaces to which you have access\.
> 
> <!-- <ul> -->
> 
> -  [Requirements and restrictions](https://dataplatform.cloud.ibm.com/docs/content/wsj/manage-data/search-assets.html?context=cdpaas&locale=en#restrictions)
> 
> -  [Searching for assets](https://dataplatform.cloud.ibm.com/docs/content/wsj/manage-data/search-assets.html?context=cdpaas&locale=en#search)
> 
> -  [Selecting results](https://dataplatform.cloud.ibm.com/docs/content/wsj/manage-data/search-assets.html?context=cdpaas&locale=en#result)
> 
> <!-- </ul> -->

Полные тексты и offsets всех 133 записей сохранены в heading_cases.jsonl. Примеры выбраны по структурным категориям, без отбора по retrieval-метрикам.

## Длины canonical-документов

Длины посчитаны закреплённым tokenizer без специальных токенов и без усечения; комментарии включены. Диапазоны выбраны после основного эксперимента: ≤510, 511–1024, 1025–2048, >2048. Они соответствуют encoder budget и кратным бюджету возвращаемого контекста. Документ длиннее 1024 токенов условно считается длинным только для этого дополнительного разреза.

| Диапазон, токены | Документы корпуса | Оценочные вопросы |
|---|---:|---:|
| <=510 | 479 | 25 |
| 511-1024 | 297 | 37 |
| 1025-2048 | 226 | 30 |
| >2048 | 142 | 26 |

Распределение по всему корпусу: min=25.0; p10=114.3; p25=284.5; p50=644.0; p75=1263.5; p90=2323.6; p95=3812.4; max=14714.0.

### Hybrid Coverage@1024 по диапазонам

| Длина | N | Fixed | Structure | Hierarchy | Без ограничений | H−S, п.п. | H−F, п.п. | H−U, п.п. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| <=510 | 25 | 0.8400 | 0.8800 | 0.9600 | 0.9200 | +8.00 | +12.00 | +4.00 |
| 511-1024 | 37 | 0.6757 | 0.7027 | 0.7027 | 0.7027 | +0.00 | +2.70 | +0.00 |
| 1025-2048 | 30 | 0.4000 | 0.5972 | 0.5972 | 0.5972 | +0.00 | +19.72 | +0.00 |
| >2048 | 26 | 0.7108 | 0.6154 | 0.7308 | 0.7308 | +11.54 | +2.00 | +0.00 |

Все retriever представлены в length_group_metrics.csv и length_group_contrasts.csv. Групповые показатели описательные; дополнительные интервалы и подтверждающие тесты не рассчитывались. Размеры групп указаны явно; данный разрез не заменяет основной результат на 118 вопросах.

## Обсуждение retrieval

У hierarchy packing BM25 Coverage@1024 = 0,8044, hybrid = 0,7366, dense = 0,6010. Гибридный поиск в выбранной конфигурации не обеспечил максимальное покрытие доказательств; влияние сегментации различалось между поисковыми компонентами. Dense у hierarchy ниже structure (0,6010 против 0,6095). Сравнение BM25 с hybrid здесь описательное, основной заранее выбранный retriever остаётся hybrid. RRF не перенастраивался.

## Проверки

Имена в aggregate_metrics_presentation.csv изменены только для представления. Значения совпадают с исходными метриками. Суммы размеров диапазонов равны 1144 документам и 118 вопросам. Взвешенные по числу вопросов групповые средние воспроизводят исходные 12 агрегатов. Хеши всех исходных артефактов проверены до и после анализа. Алгоритм не исправлялся задним числом.
