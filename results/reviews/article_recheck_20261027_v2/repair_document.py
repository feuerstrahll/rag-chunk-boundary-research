"""Apply supported wording corrections, preserving the original DOCX package.

Run with PYTHONPATH=/tmp/article-docx-deps python <this file>.
No retrieval, model inference, source-data changes, or metric changes.
"""
from pathlib import Path
from zipfile import ZipFile
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import re

from lxml import etree as E
from docx import Document

HERE = Path(__file__).resolve().parent
ARTICLE = HERE.parents[2]
TARGET = next(ARTICLE.glob('*.docx'))
EXPECTED = '30455c094df0207f02a161e2bb0499e297ffcfe3742d225b5edc3fbd00aeea81'
NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def text(node):
    return ''.join(t.text or '' for t in node.findall('.//w:t', NS))


def main():
    current = TARGET.read_bytes()
    current_sha = sha(current)
    previous = HERE / 'document_repairs.json'
    allowed = {EXPECTED}
    if previous.exists():
        allowed.add(json.loads(previous.read_text())['after_sha256'])
    if current_sha not in allowed:
        raise RuntimeError('Target changed; refusing to overwrite it')
    backup = HERE / 'before_recheck.docx'
    if backup.exists():
        assert sha(backup.read_bytes()) == EXPECTED
    else:
        assert current_sha == EXPECTED
        backup.write_bytes(current)
    with ZipFile(backup) as z:
        parts = {n: z.read(n) for n in z.namelist()}
        infos = z.infolist()
    root = E.fromstring(parts['word/document.xml'])
    body = root.find('w:body', NS)
    before_body = deepcopy(body)
    edits = []

    def replace(node, new, location, reason):
        old = text(node)
        nodes = node.findall('.//w:t', NS)
        assert nodes and old != new
        nodes[0].text = new
        for t in nodes[1:]:
            t.text = ''
        edits.append({'location': location, 'before': old, 'after': new, 'reason': reason})

    def paragraph(i, new, reason):
        assert body[i].tag == '{%s}p' % NS['w']
        replace(body[i], new, f'P{i:03}', reason)

    paragraph(29, text(body[29]).replace('воспроизводимом анализе', 'анализе по сохранённым артефактам'),
        'Do not imply an independently established end-to-end reproduction')
    paragraph(116, text(body[116]).replace('Воспроизводимый конвейер', 'Зафиксированный конвейер'),
        'Distinguish the fixed experimental pipeline from proof of reproduction')
    paragraph(113,
        'Для первоначального сравнения замороженных конфигураций подготовлена отдельная исследовательская выборка: 118 дополнительных AI-сгенерированных вопросов из 118 документов. Она не является расширенным официальным QA-разбиением IBM. Из 1 089 документов после исключения 55 ранее использованных выбраны 120 документов детерминированным чередованием страт длины, глубины разделов и структурной сложности, seed=2026. После AI-подготовки, проверки и разрешения разногласий исключены holdout_auto_114 из-за недостатка содержательного evidence и holdout_auto_118 как лицензионный материал вне принятой области технических QA. Первоначально предусмотренная проверка двумя людьми заменена отдельным AI-only протоколом; независимой человеческой верификации нет. К моменту экспериментов hierarchy и V2 вопросы и результаты уже были просмотрены, поэтому выборка является исследовательской, а не независимой подтверждающей. Оба эксперимента используют эти 118 вопросов и 147 замороженных gold spans; координатное совпадение цитат не удостоверяет полноту и достаточность разметки. Исходные 30 test-вопросов в эти retrieval-запуски не включены, но вместе с 45 train повторно используются в отдельной группе дополнительного LLM-эксперимента [A4–A5].',
        'Explicit custom-QA provenance, sampling, exclusions and departure from the two-human protocol')
    paragraph(130, text(body[130]).replace(
        'Чтобы исключить преимущество структурной стратегии над искусственно слабым baseline,',
        'Для усиления fixed-size baseline') +
        ' Fixed 480/20 подбирался по EC@5, а не по EC@1024; последующее сравнение при токенном бюджете не устанавливает превосходства над fixed baseline, отдельно настроенным для этого endpoint.',
        'Selection objective differs from the new primary endpoint')
    paragraph(152,
        'Сохранённые chunks, rankings, выбранный контекст B1024, метрики по вопросам, manifest и хеши позволяют проверить расчёты по артефактам и подготовить повторный запуск. Проверка results/validations/hierarchy_full_20261026_v1.json подтверждает пересчёт метрик и стоимости контекста; в исходном запуске все encoder inputs проверены на соблюдение предела. Проверка сохранённых артефактов не равнозначна отдельному независимому повторению всего конвейера. Дополнительный анализ заголовков и длины выполнен после основного результата без нового retrieval, изменения алгоритма и RRF. Качество генерации ответов и независимая подтверждающая оценка не проводились.',
        'Distinguish artifact replay from an independent end-to-end rerun')
    paragraph(137, text(body[137]).replace(
        'Для вопроса q эталонная информация',
        'В новых экспериментах на 118 вопросах эталонная информация для вопроса q') +
        ' В историческом результате на 75 исходных вопросах применялась прежняя агрегация: для каждого evidence выбиралось максимальное покрытие среди альтернативных alignment spans, затем покрытия отдельных evidence усреднялись внутри вопроса и вопросы получали равные веса. Исторические числа сохраняются с этим определением [A1].',
        'Restrict the length-weighted formula to the new evaluation; disclose historical aggregation')
    paragraph(161, text(body[161]).replace(
        'это единица прежней EvidenceCoverage, а не нативные токены GPT-4o.',
        'это единица бюджета B в оценке EvidenceCoverage, а не нативные токены GPT-4o; сама EvidenceCoverage измеряет долю покрытых символов gold.'),
        'Token-budget units are distinct from character-based evidence coverage')
    paragraph(162, text(body[162]).replace(
        'Единственный независимый от разметки судья —',
        'В качестве единственного LLM-судьи использован'),
        'Blinding to labels does not establish independence from their preparation')
    paragraph(189, text(body[189]) +
        ' При B1024 BM25 и dense просматривают полные доступные ранжирования, а hybrid — объединение двух списков глубины 100. Поэтому межпоисковые различия характеризуют настроенные конвейеры вместе с их пулами кандидатов и не приписываются только функции ранжирования.',
        'Retriever comparisons include different candidate-pool depths')
    paragraph(216, text(body[216]).replace(
        'Coverage@1024 и Coverage@5 совпали по каждому вопросу также для BM25 и dense (таблица 9)',
        'Для BM25 и dense каждый из показателей Coverage@1024 и Coverage@5 также совпал между V1 и обоими V2 на всех 118 вопросах (таблица 9)'),
        'Clarify equality between versions, not between different endpoints')
    paragraph(245, text(body[245]).replace(
        'Таким образом, достаточность контекста качественно дополняет EvidenceCoverage:',
        'Разобранные примеры показывают, как оценка достаточности может дополнить EvidenceCoverage:'),
        'Qualitative case evidence does not validate the metric generally')
    paragraph(264, text(body[264]).replace(
        'Самостоятельная польза иерархического ограничения не установлена.',
        'Fixed подбирался по EC@5, поэтому результат при EC@1024 не сравнивает методы, отдельно настроенные для этого endpoint. Самостоятельная польза иерархического ограничения не установлена.'),
        'Carry baseline-objective caveat into the conclusion')
    paragraph(265, text(body[265]).replace(
        'Дополнительные 58 оценок GPT-4o качественно показали возможность достаточного ответа вне размеченного evidence, но также выявили ошибки судьи;',
        'В содержательном AI-разборе отдельных оценок GPT-4o обнаружена информация для ответа вне размеченного evidence, а также оспорены некоторые решения судьи;'),
        'Attribute semantic interpretation to case review; no generated answers were evaluated')

    row = body[87].findall('w:tr', NS)[3]
    cells = row.findall('w:tc', NS)
    replace(cells[0], 'Лексическая, семантическая и модельно-ориентированная сегментация [4, 6, 21]',
            'TABLE1/R3/C0', 'TextTiling is a lexical method')
    replace(cells[1], 'Лексическая связность (TextTiling), сходство эмбеддингов либо неопределённость LLM (Meta-Chunking); в последнем — также объединение и дополнение',
            'TABLE1/R3/C1', 'Do not attribute embedding-similarity boundaries to all cited methods')
    replace(cells[4], 'Границы; в Meta-Chunking также объединение и дополнение',
            'TABLE1/R3/C4', 'Include stages beyond boundary selection')
    paragraph(90, text(body[90]).replace(
        'в семантической сегментации — близость векторных представлений соседних единиц.',
        'в методах лексической и векторной связности — тематические изменения соседних единиц. Модельно-ориентированные методы могут дополнительно использовать неопределённость языковой модели [6, 21].'),
        'Align the summary with the corrected literature classification')

    paths = {
        294: ['../techdoc-chunking-pipeline/results/watsonx_budget_matched/summary.json'],
        295: ['../techdoc-chunking-pipeline/article/results/stage_a/step3_20261018_v2/stage3_report.md', '../techdoc-chunking-pipeline/article/results/stage_a/step3_20261018_v2/summary.json'],
        296: ['../techdoc-chunking-pipeline/article/results/stage_a/step5b_mechanism_diagnostics_20261020_v1/budget_metrics.parquet', '../techdoc-chunking-pipeline/article/results/stage_a/step5b_mechanism_diagnostics_20261020_v1/mechanism_report.md'],
        297: ['../techdoc-chunking-pipeline/article/results/stage_a/step4_20261018_v2/baseline_selection.json', '../techdoc-chunking-pipeline/article/results/stage_a/step5a_holdout_preparation_v1/sampling_manifest.json', '../techdoc-chunking-pipeline/article/results/stage_a/step5a_ai_approved_freeze_v2/protocol_amendment.md'],
        298: ['../techdoc-chunking-pipeline/article/results/stage_a/step5b_20261020_v1/summary.json', '../techdoc-chunking-pipeline/article/results/stage_a/step5a_ai_approved_freeze_v2/manifest.json', '../techdoc-chunking-pipeline/article/results/stage_a/step5a_ai_approved_freeze_v2/qc_report.json'],
        299: ['../techdoc-chunking-pipeline/article/results/stage_a/step5b_exploratory_diagnostics_20261020_v1/exploratory_report.md'],
        300: ['../techdoc-chunking-pipeline/article/results/stage_a/step5b_mechanism_interpretation_20261020_v1/diagnostic_synthesis.md', '../techdoc-chunking-pipeline/article/results/stage_a/step5b_mechanism_diagnostics_20261020_v1/rank_audit.jsonl'],
        301: ['../techdoc-chunking-pipeline/article/results/stage_a/step5b_index_coverage_diagnostic_20261020_v1/index_coverage_report.md', '../techdoc-chunking-pipeline/article/results/stage_a/step5b_index_coverage_diagnostic_20261020_v1/summary.json'],
        302: ['results/runs/stage_a_train_run1_20261015/execution_cost.json'],
    }
    source_hashes = {}
    for i, files in paths.items():
        for f in files:
            p = (ARTICLE / f).resolve()
            assert p.is_file(), p
            source_hashes[str(p)] = sha(p.read_bytes())
        paragraph(i, text(body[i]) + ' Источники: ' + '; '.join(files) + '.',
                  'Concrete source locations for claims and benchmark provenance')
    paragraph(303, text(body[303]).replace(
        'тесты article — 34 passed.',
        'историческая запись проверки тестов article — 34 passed, без заявления о новом запуске тестов.'),
        'Identify a saved test record as historical')

    changed_paragraphs = {int(e['location'][1:]) for e in edits if e['location'].startswith('P')}
    for i, (a, b) in enumerate(zip(before_body, body)):
        if i not in changed_paragraphs and i != 87:
            assert E.tostring(a, method='c14n', exclusive=True) == E.tostring(b, method='c14n', exclusive=True), f'Unexpected change at body {i}'
    # All numerical result tables (3–13) must remain byte-identical at XML level.
    old_tables = before_body.findall('w:tbl', NS)
    new_tables = body.findall('w:tbl', NS)
    assert len(old_tables) == len(new_tables) == 13
    for a, b in zip(old_tables[1:], new_tables[1:]):
        assert E.tostring(a, method='c14n', exclusive=True) == E.tostring(b, method='c14n', exclusive=True)
    serialized = E.tostring(root, encoding='UTF-8', xml_declaration=True, standalone=True)
    candidate = HERE / 'checked_candidate.docx'
    with ZipFile(candidate, 'w') as z:
        for info in infos:
            z.writestr(info, serialized if info.filename == 'word/document.xml' else parts[info.filename])
    parsed = Document(candidate)
    assert len(parsed.tables) == 13
    with ZipFile(candidate) as z:
        assert z.testzip() is None
        for name, data in parts.items():
            if name != 'word/document.xml':
                assert z.read(name) == data, name
    assert sha(TARGET.read_bytes()) == current_sha
    TARGET.write_bytes(candidate.read_bytes())
    assert sha(TARGET.read_bytes()) == sha(candidate.read_bytes())
    manifest = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'target': str(TARGET), 'backup': str(backup),
        'before_sha256': EXPECTED, 'after_sha256': sha(TARGET.read_bytes()),
        'script_sha256': sha(Path(__file__).read_bytes()),
        'scope': 'Supported document wording and provenance corrections; no numerical result or research artifact changes',
        'edits': edits, 'source_sha256': source_hashes,
        'verification': {
            'package_opens_in_python_docx': True,
            'other_package_parts_unchanged': True,
            'all_12_other_tables_unchanged': True,
            'unselected_body_elements_unchanged': True,
            'visual_page_rendering': 'unavailable: Word/LibreOffice absent',
        },
    }
    (HERE / 'document_repairs.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'edits': len(edits), 'after_sha256': manifest['after_sha256'],
                      'verified_tables_unchanged': 12}, ensure_ascii=False))


if __name__ == '__main__':
    main()
