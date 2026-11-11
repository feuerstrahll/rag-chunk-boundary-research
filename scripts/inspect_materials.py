"""Build a deterministic, read-only inventory of the article workspace.

The only optional write is ``--output`` and is restricted to a new file below
``article/results/manifests``. The script never runs retrieval or downloads data.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0"
PACKAGE_NAMES = (
    "datasets", "numpy", "orjson", "pyarrow", "pydantic",
    "sentence-transformers", "tokenizers", "torch", "transformers",
)
SOURCE_FILES = (
    "pyproject.toml", "uv.lock", "configs/watsonx_retrieval.yaml",
    "configs/analysis_contract.yaml", "src/models/contracts.py",
    "src/normalization/canonical.py", "src/parsers/markdown.py",
    "src/chunking/fixed.py", "src/chunking/structure.py",
    "src/evaluation/retrieval.py", "src/retrieval/ranking.py",
    "src/ragstudy/data/watsonx_qa.py", "scripts/run_watsonx_retrieval.py",
    "scripts/run_watsonx_budget_matched.py",
)
CHUNK_FILES = (
    ("structure_chunks", "results/watsonx_retrieval/md_document/structure/chunks.jsonl"),
    ("fixed_matched_chunks", "results/watsonx_budget_matched/md_document/fixed_matched/chunks.jsonl"),
)
SUMMARY_FILE = "results/watsonx_budget_matched/summary.json"
RANKING_FILE = "results/watsonx_retrieval/md_document/structure/rankings.jsonl"


def utc_timestamp(now: datetime | None = None) -> str:
    value = now or datetime.now(timezone.utc)
    if value.tzinfo is None:
        raise ValueError("Inventory timestamp must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_info(path: Path, root: Path, *, file_id: str | None = None, role: str | None = None) -> dict[str, Any]:
    """Return deterministic metadata for one existing regular file."""
    return {
        "id": file_id or path.name,
        "role": role or "unspecified",
        "path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _issue(code: str, message: str, path: str | None = None) -> dict[str, str]:
    issue = {"code": code, "message": message}
    if path is not None:
        issue["path"] = path
    return issue


def _git(repository: Path, *, scope: Path | None = None) -> dict[str, Any]:
    base = [
        "git",
        "-c",
        f"safe.directory={repository.as_posix()}",
        "-c",
        "core.quotepath=false",
    ]

    def run(*arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [*base, *arguments],
            cwd=repository,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )

    root = run("rev-parse", "--show-toplevel")
    head = run("rev-parse", "HEAD")
    status_arguments = ["status", "--short", "--untracked-files=all"]
    if scope is not None:
        status_arguments.extend(["--", str(scope)])
    status = run(*status_arguments)
    errors = sorted(
        message.strip()
        for completed in (root, status)
        for message in completed.stderr.splitlines()
        if message.strip() and "unable to access" not in message
    )
    lines = sorted(line.rstrip() for line in status.stdout.splitlines() if line.strip())
    return {
        "path": str(repository),
        "git_root": root.stdout.strip() if root.returncode == 0 else None,
        "head": head.stdout.strip() if head.returncode == 0 else None,
        "unborn_head": head.returncode != 0 and root.returncode == 0,
        "dirty": bool(lines) if status.returncode == 0 else None,
        "status": lines,
        "errors": errors,
    }


def _environment() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in PACKAGE_NAMES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "packages": packages,
    }


def _load_pyarrow() -> Any | None:
    try:
        from pyarrow import parquet
    except ImportError:
        return None
    return parquet


def _parquet_record(path: Path, root: Path, *, file_id: str, parquet: Any | None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "id": file_id,
        "path": path.relative_to(root).as_posix(),
        "metadata_available": False,
        "row_count": None,
        "row_group_count": None,
        "schema": None,
    }
    if parquet is None or not path.is_file():
        return record
    parquet_file = parquet.ParquetFile(path)
    record.update(
        metadata_available=True,
        row_count=parquet_file.metadata.num_rows,
        row_group_count=parquet_file.metadata.num_row_groups,
        schema=[
            {"name": field.name, "type": str(field.type), "nullable": field.nullable}
            for field in parquet_file.schema_arrow
        ],
    )
    return record


def _chunk_statistics(path: Path) -> dict[str, int]:
    count = tokens = 0
    documents: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            chunk = json.loads(line)
            count += 1
            tokens += int(chunk["token_count"])
            documents.add(str(chunk["doc_id"]))
    return {"chunks": count, "tokens": tokens, "documents": len(documents)}


def _ranking_statistics(path: Path) -> dict[str, Any]:
    queries: dict[str, str] = {}
    methods: Counter[str] = Counter()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            queries[str(row["query_id"])] = str(row["split"])
            methods[str(row["method"])] += 1
    return {
        "unique_questions": len(queries),
        "questions_by_split": dict(sorted(Counter(queries.values()).items())),
        "rows_by_method": dict(sorted(methods.items())),
    }


def build_inventory(
    article_root: Path,
    upstream_root: Path,
    *,
    now: datetime | None = None,
    parquet_module: Any = ...,
) -> dict[str, Any]:
    article_root = article_root.resolve()
    workspace_root = article_root.parent.resolve()
    upstream_root = upstream_root.resolve()
    warnings: list[dict[str, str]] = []
    errors: list[dict[str, str]] = []
    parquet = _load_pyarrow() if parquet_module is ... else parquet_module
    if parquet is None:
        warnings.append(_issue("pyarrow_unavailable", "Parquet row counts and schemas were not inspected."))

    files: list[dict[str, Any]] = []

    def add_file(file_id: str, relative: str, role: str) -> Path | None:
        path = upstream_root / relative
        if not path.is_file():
            errors.append(_issue("missing_file", "Required upstream file is missing.", relative))
            return None
        files.append(file_info(path, upstream_root, file_id=file_id, role=role))
        return path

    for relative in SOURCE_FILES:
        add_file(f"source:{relative}", relative, "upstream_source")

    summary_path = add_file("historical_summary", SUMMARY_FILE, "historical_result")
    summary: dict[str, Any] | None = None
    if summary_path is not None:
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(_issue("invalid_summary", str(exc), SUMMARY_FILE))

    parquet_records: list[dict[str, Any]] = []
    if summary is not None:
        revision = str(summary.get("dataset_revision", ""))
        input_specs = (
            ("corpus", "corpus/train-00000-of-00001.parquet", summary.get("corpus_sha256")),
            ("qa_train", "question_answers/train-00000-of-00001.parquet", summary.get("qa_sha256", {}).get("train")),
            ("qa_test", "question_answers/test-00000-of-00001.parquet", summary.get("qa_sha256", {}).get("test")),
        )
        for file_id, suffix, expected_hash in input_specs:
            relative = f"data/raw/watsonx_docsqa/{revision}/{suffix}"
            path = upstream_root / relative
            if not path.is_file():
                errors.append(_issue("missing_file", "Required upstream file is missing.", relative))
                parquet_records.append(_parquet_record(path, upstream_root, file_id=file_id, parquet=parquet))
                continue
            try:
                parquet_records.append(_parquet_record(path, upstream_root, file_id=file_id, parquet=parquet))
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                errors.append(_issue("parquet_metadata_error", str(exc), relative))
                parquet_records.append(_parquet_record(path, upstream_root, file_id=file_id, parquet=None))
            add_file(file_id, relative, "raw_input")
            actual_hash = files[-1]["sha256"]
            files[-1]["expected_sha256"] = expected_hash
            files[-1]["matches_expected_sha256"] = actual_hash == expected_hash
            if actual_hash != expected_hash:
                errors.append(_issue("hash_mismatch", "Raw input SHA-256 differs from historical summary.", relative))

    chunk_sets: list[dict[str, Any]] = []
    for file_id, relative in CHUNK_FILES:
        path = add_file(file_id, relative, "historical_chunks")
        if path is not None:
            try:
                chunk_sets.append({"id": file_id, **_chunk_statistics(path)})
            except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
                errors.append(_issue("invalid_chunks", str(exc), relative))

    ranking_path = add_file("structure_rankings", RANKING_FILE, "historical_rankings")
    ranking: dict[str, Any] | None = None
    if ranking_path is not None:
        try:
            ranking = _ranking_statistics(ranking_path)
        except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
            errors.append(_issue("invalid_rankings", str(exc), RANKING_FILE))

    historical_result: dict[str, Any] | None = None
    if summary is not None:
        try:
            historical_result = {
                "status": "historical_not_reproduced",
                "documents": summary["documents"],
                "questions": summary["questions"],
                "seed": summary["seed"],
                "dataset_revision": summary["dataset_revision"],
                "embedding": summary["embedding"],
                "resolved_text_budget": summary["resolved_text_budget"],
                "selected_fixed": summary["budget_matching"]["selected"],
                "primary_result": summary["primary_result"],
                "chunk_sets": sorted(chunk_sets, key=lambda item: item["id"]),
                "structure_rankings": ranking,
            }
        except KeyError as exc:
            errors.append(_issue("invalid_summary_schema", f"Missing key: {exc}", SUMMARY_FILE))

    article_scope = article_root.relative_to(workspace_root)
    repositories = {
        "article": _git(workspace_root, scope=article_scope),
        "root": _git(workspace_root),
        "upstream": _git(upstream_root),
    }
    repositories["article"]["path"] = str(article_root)
    repositories["article"]["containing_repository"] = str(workspace_root)
    for name, record in repositories.items():
        for message in record["errors"]:
            warnings.append(_issue("git_inspection_warning", f"{name}: {message}"))

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": utc_timestamp(now),
        "environment": _environment(),
        "repositories": repositories,
        "files": sorted(files, key=lambda item: item["id"]),
        "parquet": sorted(parquet_records, key=lambda item: item["id"]),
        "historical_result": historical_result,
        "warnings": sorted(warnings, key=lambda item: (item["code"], item.get("path", ""), item["message"])),
        "errors": sorted(errors, key=lambda item: (item["code"], item.get("path", ""), item["message"])),
    }


def inspect(root: Path) -> dict[str, Any]:
    """Compatibility wrapper for the original upstream-root API."""
    article_root = Path(__file__).resolve().parents[1]
    return build_inventory(article_root, root)


def _validated_output(path: Path, article_root: Path) -> Path:
    output = path.resolve()
    allowed = (article_root / "results" / "manifests").resolve()
    if output == allowed or allowed not in output.parents:
        raise ValueError(f"--output must be a file inside {allowed}")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing manifest: {output}")
    return output


def main() -> int:
    article_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root", type=Path,
        default=article_root.parent / "techdoc-chunking-pipeline",
        help="read-only upstream pipeline checkout",
    )
    parser.add_argument("--output", type=Path, help="new JSON file below article/results/manifests")
    arguments = parser.parse_args()
    result = build_inventory(article_root, arguments.source_root)
    rendered = json.dumps(result, ensure_ascii=True, indent=2) + "\n"
    if arguments.output is not None:
        output = _validated_output(arguments.output, article_root)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")
    return 1 if result["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
