from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ARTICLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ARTICLE / "scripts"))


def load_script(name: str):
    path = ARTICLE / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


baseline = load_script("run_stage_a_baseline.py")
snapshot = load_script("prepare_baseline_snapshot.py")


def test_stage_a_paths_exclude_historical_test(tmp_path: Path) -> None:
    dataset = {"corpus": "corpus.parquet", "train_qa": "train.parquet", "historical_test_qa": "test.parquet"}
    paths = baseline.resolve_input_paths(tmp_path, dataset, "stage_a_train")
    assert set(paths) == {"corpus", "train_qa"}
    assert all("test.parquet" not in str(path) for path in paths.values())


def test_historical_mode_requires_explicit_permission() -> None:
    with pytest.raises(PermissionError, match="--allow-historical-regression"):
        baseline.validate_mode("historical_regression", False)
    baseline.validate_mode("historical_regression", True)


def test_run_directory_must_be_new_and_inside_results(tmp_path: Path) -> None:
    article = tmp_path / "article"
    valid = article / "results" / "runs" / "run-1"
    assert baseline.validate_run_dir(valid, article) == valid.resolve()
    with pytest.raises(ValueError):
        baseline.validate_run_dir(article / "elsewhere", article)
    valid.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        baseline.validate_run_dir(valid, article)


def test_compare_runs_detects_identical_and_changed_artifacts(tmp_path: Path) -> None:
    current, reference = tmp_path / "current", tmp_path / "reference"
    current.mkdir()
    reference.mkdir()
    names = ("chunks.parquet", "retrieval_runs.parquet", "index_statistics.json")
    for name in names:
        (current / name).write_bytes(name.encode())
        (reference / name).write_bytes(name.encode())
    assert baseline.compare_runs(current, reference)["status"] == "identical"
    (current / "chunks.parquet").write_bytes(b"changed")
    assert baseline.compare_runs(current, reference)["status"] == "different"


def test_snapshot_validation_detects_modified_file(tmp_path: Path) -> None:
    root = tmp_path / "snapshot-id"
    source = root / "src" / "module.py"
    source.parent.mkdir(parents=True)
    source.write_text("value = 1\n", encoding="utf-8")
    records = [{"path": "src/module.py", "size_bytes": source.stat().st_size, "sha256": snapshot.sha256_file(source)}]
    identity = snapshot.snapshot_id(records)
    target = tmp_path / identity
    root.rename(target)
    (target / "source_snapshot_manifest.json").write_text(
        json.dumps({"snapshot_id": identity, "files": records}), encoding="utf-8"
    )
    snapshot.validate_snapshot(target)
    (target / "src" / "module.py").write_text("value = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Snapshot file changed"):
        snapshot.validate_snapshot(target)
