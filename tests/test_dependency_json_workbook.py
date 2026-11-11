"""The JSON workbook must preserve empty decisions and grouped canonical A spans."""
import json
from pathlib import Path

import pytest

from scripts.dependency_json_workbook import create, export


def test_unlabeled_workbook_and_group_export(tmp_path: Path) -> None:
    source = tmp_path / "unlabeled.json"
    create(source)
    workbook = json.loads(source.read_text(encoding="utf-8"))
    assert len(workbook["records"]) == 54
    assert all(not row["answer"]["decision"] for row in workbook["records"])
    with pytest.raises(ValueError, match="Incomplete records"):
        export(source, tmp_path / "empty.jsonl", allow_incomplete=False)

    row = workbook["records"][0]
    blocks = workbook["documents"][row["doc_id"]]["blocks"]
    a_blocks = blocks[0:2]
    answer = row["answer"]
    answer.update(decision="positive", relation_type="other", missing_context="test context",
                  rationale="test only", annotator_id="test_fixture", dependency_group="test_group",
                  necessary_a=[{"block_id": b["block_id"], "span": [b["char_start"], b["char_end"]]}
                               for b in a_blocks])
    edited = tmp_path / "edited.json"
    edited.write_text(json.dumps(workbook, ensure_ascii=False), encoding="utf-8")
    labels = tmp_path / "labels.jsonl"
    export(edited, labels, allow_incomplete=True)
    edges = [json.loads(line) for line in labels.read_text(encoding="utf-8").splitlines()]
    assert len(edges) == 2
    assert {edge["edge_id"] for edge in edges} == {row["record_id"] + ":1", row["record_id"] + ":2"}
    assert all(edge["dependency_group"] == "test_group" for edge in edges)
    assert all(edge["review_status"] == "draft" for edge in edges)
