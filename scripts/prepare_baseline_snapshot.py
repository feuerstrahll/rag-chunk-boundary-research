"""Create or validate a content-addressed, read-only upstream source snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DIRECTORIES = (
    "src/chunking",
    "src/enrichment",
    "src/evaluation",
    "src/models",
    "src/normalization",
    "src/parsers",
    "src/ragstudy/data",
    "src/retrieval",
)
FILES = (
    "src/ragstudy/__init__.py",
    "pyproject.toml",
    "uv.lock",
    "configs/watsonx_retrieval.yaml",
    "configs/analysis_contract.yaml",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_files(upstream: Path) -> list[Path]:
    paths = [upstream / relative for relative in FILES]
    for directory in DIRECTORIES:
        paths.extend((upstream / directory).glob("*.py"))
    return sorted((path for path in paths if path.is_file()), key=lambda path: path.relative_to(upstream).as_posix())


def git_state(upstream: Path) -> dict[str, Any]:
    base = ["git", "-c", f"safe.directory={upstream.as_posix()}", "-c", "core.quotepath=false"]

    def run(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [*base, *args], cwd=upstream, capture_output=True, text=True,
            encoding="utf-8", errors="replace", check=False,
        )

    head = run("rev-parse", "HEAD")
    status = run("status", "--short", "--untracked-files=all")
    diff = run("diff", "--binary", "HEAD", "--", *(str(path.relative_to(upstream)) for path in selected_files(upstream)))
    if head.returncode or status.returncode or diff.returncode:
        raise RuntimeError("Unable to record upstream Git state")
    return {
        "head": head.stdout.strip(),
        "dirty": bool(status.stdout.strip()),
        "status": sorted(line.rstrip() for line in status.stdout.splitlines() if line.strip()),
        "selected_tracked_diff_sha256": hashlib.sha256(diff.stdout.encode("utf-8")).hexdigest(),
    }


def source_records(upstream: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(upstream).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in selected_files(upstream)
    ]


def snapshot_id(records: list[dict[str, Any]]) -> str:
    identity = json.dumps(records, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(identity).hexdigest()


def create_snapshot(article: Path, upstream: Path) -> Path:
    article, upstream = article.resolve(), upstream.resolve()
    records = source_records(upstream)
    identity = snapshot_id(records)
    destination = article / "data" / "intermediate" / "upstream_snapshot" / identity
    manifest_path = destination / "source_snapshot_manifest.json"
    if destination.exists():
        if not manifest_path.is_file():
            raise FileExistsError(f"Incomplete snapshot directory already exists: {destination}")
        validate_snapshot(destination)
        return destination
    destination.mkdir(parents=True)
    for record in records:
        source = upstream / record["path"]
        target = destination / record["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    manifest = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "snapshot_id": identity,
        "classification": "current_upstream_snapshot_not_historical_source",
        "upstream_path": str(upstream),
        "git": git_state(upstream),
        "files": records,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8", newline="\n")
    validate_snapshot(destination)
    return destination


def validate_snapshot(snapshot: Path) -> dict[str, Any]:
    manifest_path = snapshot / "source_snapshot_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["snapshot_id"] != snapshot.name:
        raise ValueError("Snapshot directory does not match manifest identity")
    actual_records = []
    for expected in manifest["files"]:
        path = snapshot / expected["path"]
        if not path.is_file():
            raise FileNotFoundError(f"Snapshot file missing: {expected['path']}")
        actual = {"path": expected["path"], "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        if actual != expected:
            raise ValueError(f"Snapshot file changed: {expected['path']}")
        actual_records.append(actual)
    if snapshot_id(actual_records) != manifest["snapshot_id"]:
        raise ValueError("Snapshot aggregate identity changed")
    return manifest


def main() -> None:
    article = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, default=article.parent / "techdoc-chunking-pipeline")
    args = parser.parse_args()
    destination = create_snapshot(article, args.upstream)
    print(destination)


if __name__ == "__main__":
    main()
