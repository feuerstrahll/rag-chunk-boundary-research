# Workspace context

Verified by read-only preflight on 15.10.2026. This document records the working boundary for Stage A; it does not claim that the historical experiment is reproducible.

## Repository boundary

`C:\Users\nasty\ДИПЛОМ\article` is the writable article workspace. It currently sits inside the parent Git repository `C:\Users\nasty\ДИПЛОМ` and is reported as untracked (`?? article/`). It does not have its own `.git` directory.

`C:\Users\nasty\ДИПЛОМ\techdoc-chunking-pipeline` is a separate upstream Git repository. Its current HEAD is `7d986ef68401b475f3cdc33694ac0c8cc7bac630`, and its working tree contains pre-existing tracked and untracked changes. It and its historical results are read-only for article work.

The historical manifests were themselves produced from dirty trees: the structure manifest records commit `87fd0d77150bbdefb4b2f9613f334c2b2c24fbb8`; the budget-matched fixed manifest records `912cecb1b3527fc665cd1d09803c09867895e06d`. Neither manifest contains the corresponding uncommitted patch. Therefore HEAD, manifest metadata, and the present checkout are insufficient to reconstruct the exact historical source state.

## Directory contract

| Path | Intended content | Policy |
| --- | --- | --- |
| `configs/` | Frozen experiment, data, and analysis contracts | Versioned; parameters fixed before confirmatory runs |
| `src/` | Article-owned segmentation and diagnostic modules | No implicit writes to upstream |
| `tests/` | Unit/integrity/determinism/leakage checks | Must avoid model downloads and full retrieval by default |
| `data/raw/` | Immutable pinned inputs or manifests pointing to them | Never rewrite; record hashes |
| `data/intermediate/` | Regenerable canonical documents, alignments, labels, and diagnostics | Trace every file to config and raw inputs |
| `results/` | Run-specific metrics, manifests, reports, and logs | Never overwrite historical or completed runs |

These directories are a contract, not evidence that corresponding artifacts already exist. Create them only when required by an authorized task.

## Read-only upstream inputs

The Stage A source corpus is the Markdown column of IBM watsonxDocsQA revision `f7da4b6d8a8c4609c49fda06af3bea29b6ef9be4`.

| Upstream file | Rows | SHA-256 |
| --- | ---: | --- |
| `data/raw/watsonx_docsqa/<revision>/corpus/train-00000-of-00001.parquet` | 1,144 | `fb56d7c847f576d72f32669b01feb47e38788b7e91deeecdcdbcc2b9995fdd6b` |
| `data/raw/watsonx_docsqa/<revision>/question_answers/train-00000-of-00001.parquet` | 45 | `73c81739e29b2e5fac82accb43d7b3a7f319e047ae56f421dd8d3804f0742bb1` |
| `data/raw/watsonx_docsqa/<revision>/question_answers/test-00000-of-00001.parquet` | 30 | `bea3bd567eb95b29b3d3e4b185ed6bb0bad114b897ac42c9bac3f4fec054a18a` |

Treat all three files as immutable. The QA fields are `question_id`, `question`, `correct_answer`, `correct_answer_document_ids`, and `ground_truths_contexts`; the final two are stored as strings and may encode JSON arrays.

Relevant upstream implementation and artifacts:

- `src/chunking/structure.py`: `StructureAwareChunker`.
- `src/parsers/markdown.py` and `src/normalization/canonical.py`: Markdown blocks and canonical offsets.
- `src/retrieval/ranking.py`: BM25, dense retrieval, chunk-first RRF, and document projection.
- `src/evaluation/retrieval.py`: evidence alignment, coverage, integrity statistics, and paired bootstrap.
- `scripts/run_watsonx_retrieval.py`: fixed/structure retrieval entrypoint.
- `scripts/run_watsonx_budget_matched.py`: Markdown corpus-only budget matching entrypoint.
- `results/watsonx_retrieval/md_document/structure/{chunks.jsonl,rankings.jsonl,manifest.json}`.
- `results/watsonx_budget_matched/summary.json`.

The stored budget-matched summary reports Hybrid `EvidenceCoverage@5` 0.6591693575 for fixed and 0.8802656941 for structure-aware, difference +0.2210963367 with 95% CI [0.1378109009, 0.3103155149]. This is a verified historical artifact, not a reproduced article result.

## Environment observed during preflight

- Upstream virtual environment: `..\techdoc-chunking-pipeline\.venv`.
- Python 3.11.2; uv 0.12.21.
- Key packages: datasets 5.0.1, pyarrow 25.0.1, sentence-transformers 6.1.0, transformers 5.18.0, tokenizers 0.23.2, torch 2.14.1, numpy 2.4.6, pydantic 2.13.5.
- Dense model: `sentence-transformers/multi-qa-MiniLM-L6-cos-v1`, revision `9c441fc9d77eb051358e49979a385d07a511c3c5`; the exact snapshot was present locally during preflight.
- Historical runtime text budget: 510 tokens. Resolve and validate it again during a future rerun.

This environment belongs to upstream, and its `pyproject.toml` and `uv.lock` are currently modified. It may be used for read-only inspection; an article experiment needs a frozen environment or a fully recorded upstream snapshot.

## Safe inspection and future commands

Read-only inventory from `article`:

```powershell
& ..\techdoc-chunking-pipeline\.venv\Scripts\python.exe .\scripts\inspect_materials.py
```

The script reads hashes, manifests, counts, Git status, and selected artifacts. It does not rerun parsing, embeddings, ranking, evaluation, or bootstrap.

Upstream provides these experiment entrypoints, but they were not executed during preflight:

```powershell
cd C:\Users\nasty\ДИПЛОМ\techdoc-chunking-pipeline
uv run python scripts/run_watsonx_retrieval.py --smoke --output <new-output-path>
uv run python scripts/run_watsonx_retrieval.py --output <new-output-path>
uv run python scripts/run_watsonx_budget_matched.py --smoke --output <new-output-path>
uv run python scripts/run_watsonx_budget_matched.py --output <new-output-path>
```

Do not execute them until source state, environment, input hashes, offline model availability, output location, and leakage protocol are frozen. The first retrieval entrypoint does not force `local_files_only`; the shared data resolver can download missing Parquet files. A controlled rerun must fail closed rather than download implicitly.

## Reproducibility and leakage status

No end-to-end rerun has been completed for the article. Reading summaries, matching input hashes, finding the model cache, and running `inspect_materials.py` do not establish reproducibility.

All 75 questions have already appeared in historical analysis. Use the 45 original train questions for Stage A development diagnostics. Do not treat the 30 rows named `test` as a fresh independent holdout. Gold evidence may support diagnostic oracle analysis, but it must not become a CBO input, tuning target, or source of final-test decisions.

Before a baseline can be called reproduced, a frozen snapshot must regenerate canonical documents, chunks, embeddings, rankings, per-question metrics, aggregate metrics, and paired bootstrap outputs into a new result directory, with provenance and output hashes. Any mismatch must be localized before proceeding to CBO.

## Open decisions before Stage A execution

- How the exact upstream source state will be frozen despite dirty historical manifests.
- Whether article code will use an explicit adapter, a packaged dependency, or exported immutable artifacts.
- Where the independent final evaluation set will come from.
- The formal oracle constraints and shared retrieved-token budget.
- The run-directory naming and manifest schema for article-owned results.
