"""Apply the ten-case Codex content review to the supplied AI draft workbook.

This is a substantive AI review, not human or independent verification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SOURCE_SHA256 = "ec074bc6864b35630f7e8c1b323a106ba8a3fc0b5c2cf3e0630bbf6497192ddc"
REVIEWED = {"pilot_001", "pilot_005", "pilot_008", "pilot_014", "pilot_025",
            "pilot_031", "pilot_032", "pilot_033", "pilot_039", "pilot_047"}


def apply(source: Path, output: Path) -> None:
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError("The supplied AI draft differs from the reviewed input")
    workbook = json.loads(raw)
    records = {row["record_id"]: row for row in workbook["records"]}
    if len(records) != 54 or not REVIEWED <= records.keys():
        raise ValueError("Expected 54 pilot records and all ten review cases")

    for record_id in REVIEWED:
        answer = records[record_id]["answer"]
        if not answer["decision"] or answer["review_status"] != "draft":
            raise ValueError(f"Unexpected initial annotation: {record_id}")
        answer["review_status"] = "reviewed_by_codex"
        answer["reviewer_id"] = "AI_Codex"
        answer["review_scope"] = "content, minimal A, canonical coordinates, and existing chunks"

    for record_id in ("pilot_031", "pilot_032", "pilot_033"):
        answer = records[record_id]["answer"]
        answer.update(
            missing_context="Не требуется для навигационной функции пункта со ссылкой.",
            decision="negative",
            relation_type="none",
            necessary_a=[],
            dependency_group="",
            rationale=(
                "B является пунктом оглавления: текст называет действие, а URL ведёт на "
                "конкретный якорь страницы data_flows.html. Ближайшая группа «Working with the "
                "data sets» помогает ориентироваться, соседние пункты называют Data Refinery flow, "
                "но ни один из них не требуется для понимания назначения ссылки. Дальний корневой "
                "заголовок тем более не является обязательным A. Описание самой операции нужно "
                "искать на целевой странице, а не подменять заголовком текущего оглавления. "
                "Три записи относятся к одной повторяющейся ситуации этого оглавления."
            ),
            simple_structural_rule="none",
            local_pattern="self_contained_navigation_link",
            external_anchor="link_target_for_procedure_not_required_for_navigation",
        )

    answer = records["pilot_047"]["answer"]
    answer.update(
        missing_context="Не требуется для понимания вводной как указателя на следующий список.",
        decision="negative",
        relation_type="none",
        necessary_a=[],
        dependency_group="",
        rationale=(
            "B задаёт тему и обещает следующие примеры механизмов; слово common не означает "
            "исчерпывающий перечень. Ни один из семи пунктов не обязателен для понимания самой "
            "вводной, и пункты не требуют друг друга. Для ответа на вопрос о конкретных "
            "механизмах нужны соответствующие пункты, а для интерпретации отдельных пунктов "
            "может быть полезна вводная или заголовок Common security mechanisms. "
            "Целостность списка — отдельное структурное требование, а не группа B→семь A."
        ),
        simple_structural_rule="Для структурной целостности связывать вводную с последовательным списком.",
        local_pattern="intro_list_structural_unit_only",
        external_anchor="none",
    )

    workbook["status"] = "ai_draft_54_with_codex_content_review_10"
    workbook["review_note"] = (
        "Ten records received substantive Codex review. This is AI review of an AI draft; "
        "no record is independently or human verified. Original input SHA-256: " + SOURCE_SHA256
    )
    output.write_text(json.dumps(workbook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Updated {len(REVIEWED)} of {len(records)} records: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    apply(args.source, args.output)


if __name__ == "__main__":
    main()
