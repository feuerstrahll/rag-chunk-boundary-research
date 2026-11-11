from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "inspect_materials.py"
SPEC = importlib.util.spec_from_file_location("inspect_materials", SCRIPT)
assert SPEC and SPEC.loader
inspect_materials = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(inspect_materials)

FIXED_TIME = datetime(2026, 10, 15, 12, 0, tzinfo=timezone.utc)


def test_missing_files_are_explicit_errors(tmp_path: Path) -> None:
    article = tmp_path / "article"
    upstream = tmp_path / "upstream"
    article.mkdir()
    upstream.mkdir()
    manifest = inspect_materials.build_inventory(article, upstream, now=FIXED_TIME, parquet_module=None)
    missing = [item for item in manifest["errors"] if item["code"] == "missing_file"]
    assert missing
    assert any(item.get("path") == inspect_materials.SUMMARY_FILE for item in missing)
    assert manifest["historical_result"] is None


def test_sha256_file_hashes_bytes(tmp_path: Path) -> None:
    sample = tmp_path / "sample.bin"
    sample.write_bytes(b"abc")
    assert inspect_materials.sha256_file(sample) == hashlib.sha256(b"abc").hexdigest()
    assert inspect_materials.file_info(sample, tmp_path, file_id="sample", role="test") == {
        "id": "sample", "role": "test", "path": "sample.bin", "size_bytes": 3,
        "sha256": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
    }


def test_pyarrow_unavailable_has_warning_and_no_false_counts(tmp_path: Path) -> None:
    article = tmp_path / "article"
    upstream = tmp_path / "upstream"
    article.mkdir()
    upstream.mkdir()
    summary_path = upstream / inspect_materials.SUMMARY_FILE
    summary_path.parent.mkdir(parents=True)
    summary_path.write_text(json.dumps({
        "documents": 0, "questions": 0, "seed": 2026, "dataset_revision": "revision",
        "embedding": {}, "resolved_text_budget": 510, "corpus_sha256": "missing",
        "qa_sha256": {"train": "missing", "test": "missing"},
        "budget_matching": {"selected": {}}, "primary_result": {},
    }), encoding="utf-8")
    manifest = inspect_materials.build_inventory(article, upstream, now=FIXED_TIME, parquet_module=None)
    assert any(item["code"] == "pyarrow_unavailable" for item in manifest["warnings"])
    assert len(manifest["parquet"]) == 3
    assert all(item["metadata_available"] is False for item in manifest["parquet"])
    assert all(item["row_count"] is None for item in manifest["parquet"])
    assert all(item["schema"] is None for item in manifest["parquet"])


def test_parquet_metadata_is_inspected_before_full_file_hash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    article = tmp_path / "article"
    upstream = tmp_path / "upstream"
    article.mkdir()
    upstream.mkdir()
    revision = "revision"
    summary_path = upstream / inspect_materials.SUMMARY_FILE
    summary_path.parent.mkdir(parents=True)
    summary_path.write_text(json.dumps({
        "documents": 0, "questions": 0, "seed": 2026, "dataset_revision": revision,
        "embedding": {}, "resolved_text_budget": 510, "corpus_sha256": "expected",
        "qa_sha256": {"train": "expected", "test": "expected"},
        "budget_matching": {"selected": {}}, "primary_result": {},
    }), encoding="utf-8")
    raw_paths = [
        upstream / f"data/raw/watsonx_docsqa/{revision}/corpus/train-00000-of-00001.parquet",
        upstream / f"data/raw/watsonx_docsqa/{revision}/question_answers/train-00000-of-00001.parquet",
        upstream / f"data/raw/watsonx_docsqa/{revision}/question_answers/test-00000-of-00001.parquet",
    ]
    for path in raw_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"parquet-placeholder")
    events: list[str] = []
    original_hash = inspect_materials.sha256_file

    def tracked_hash(path: Path) -> str:
        if path in raw_paths:
            events.append(f"hash:{path.name}:{path.parent.name}")
        return original_hash(path)

    def tracked_metadata(path: Path, root: Path, *, file_id: str, parquet: object) -> dict:
        events.append(f"metadata:{file_id}")
        return {"id": file_id, "path": path.relative_to(root).as_posix(), "metadata_available": True,
                "row_count": 0, "row_group_count": 0, "schema": []}

    monkeypatch.setattr(inspect_materials, "sha256_file", tracked_hash)
    monkeypatch.setattr(inspect_materials, "_parquet_record", tracked_metadata)
    inspect_materials.build_inventory(article, upstream, now=FIXED_TIME, parquet_module=object())

    for file_id, path in zip(("corpus", "qa_train", "qa_test"), raw_paths):
        assert events.index(f"metadata:{file_id}") < events.index(f"hash:{path.name}:{path.parent.name}")


def test_stable_top_level_schema_and_serialization(tmp_path: Path) -> None:
    article = tmp_path / "article"
    upstream = tmp_path / "upstream"
    article.mkdir()
    upstream.mkdir()
    first = inspect_materials.build_inventory(article, upstream, now=FIXED_TIME, parquet_module=None)
    second = inspect_materials.build_inventory(article, upstream, now=FIXED_TIME, parquet_module=None)
    assert list(first) == [
        "schema_version", "generated_at_utc", "environment", "repositories", "files",
        "parquet", "historical_result", "warnings", "errors",
    ]
    assert first["schema_version"] == "1.0"
    assert first["generated_at_utc"] == "2026-10-15T12:00:00Z"
    assert json.dumps(first, ensure_ascii=True, indent=2) == json.dumps(second, ensure_ascii=True, indent=2)


def test_output_must_be_new_file_inside_manifest_directory(tmp_path: Path) -> None:
    article = tmp_path / "article"
    article.mkdir()
    allowed = article / "results" / "manifests" / "inventory.json"
    assert inspect_materials._validated_output(allowed, article) == allowed.resolve()
    with pytest.raises(ValueError):
        inspect_materials._validated_output(article / "inventory.json", article)
    allowed.parent.mkdir(parents=True)
    allowed.write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError):
        inspect_materials._validated_output(allowed, article)


def test_git_inspection_handles_unicode_repository_path(tmp_path: Path) -> None:
    repository = tmp_path / "репозиторий"
    repository.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)

    state = inspect_materials._git(repository)

    assert state["git_root"] is not None
    assert "репозиторий" in state["git_root"]
    assert "�" not in state["git_root"]
    assert state["head"] is None
    assert state["unborn_head"] is True
