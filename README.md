# Chunk boundaries in technical-document RAG

Research code and saved experiments for **context-dependent segmentation of long technical documents**. The study asks how the choice of chunk boundaries affects retrieval of source evidence when the document corpus, parser, embedding model, and retrievers are held fixed.

## Main findings

The table reports **hybrid retrieval** on 118 additional, AI-generated questions from previously unused documents. All methods searched the same 1,144-document Markdown corpus. Higher is better; the two columns impose different limits on the returned context.

| Chunking method | EvidenceCoverage@5 | EvidenceCoverage at 1,024 tokens |
| --- | ---: | ---: |
| Fixed-size, 480 tokens / 20 overlap | **0.852** | 0.648 |
| Structure-aware | 0.780 | 0.694 |
| Hierarchical packing, V1 | 0.839 | **0.737** |
| Hierarchical packing, V2 heading repair | 0.839 | **0.737** |

- **The evaluation limit changes the comparison.** Fixed-size exceeded structure-aware by 7.19 percentage points at top-5 (95% paired interval for structure minus fixed: −14.37 to −0.41 points). Under the 1,024-token limit, structure-aware exceeded fixed-size by 4.61 points, but the 95% interval (−3.63 to +13.05 points) includes zero. The fixed configuration was selected on development questions for the top-5 endpoint, not separately tuned for the token-budget endpoint.
- **Hierarchy remains exploratory.** V1 exceeded structure-aware by 4.24 points at 1,024 tokens; its descriptive 95% interval (−1.69 to +10.17 points) includes zero. Its difference from the unrestricted packing ablation was also uncertain. These comparisons do not establish a unique retrieval benefit from the hierarchy constraint.
- **Structural integrity improved without measured coverage gain.** V2 repaired all 132 previously identified heading–body separations and preserved 4,339/4,339 heading–body links across the corpus. On these 118 questions, EvidenceCoverage@5 and at 1,024 tokens were unchanged for all three retrievers.

EvidenceCoverage is the fraction of aligned gold-evidence characters covered by retrieved chunks, averaged over questions. It measures **retrieval coverage**, not answer accuracy or whether the context is sufficient to generate an answer. The 118-question gold set received AI-only review; independent human validation and evaluation on a new corpus remain open.

See the [results guide](docs/results_overview.md) for the experiment design, comparisons, limitations, and links to saved artifacts.

## Repository map

| Path | Contents |
| --- | --- |
| [`src/article_research/`](src/article_research/) | Article-owned packing and evaluation code |
| [`scripts/`](scripts/) and [`configs/`](configs/) | Experiment entry points and frozen configurations |
| [`results/runs/`](results/runs/) | Run manifests, per-question metrics, rankings, and reports |
| [`results/analyses/`](results/analyses/) | Diagnostic and post hoc analyses |
| [`docs/`](docs/) | Research protocols, workspace context, and result interpretation |
| [`tests/`](tests/) | Correctness and integrity checks |

The source corpus and model weights are not bundled. Some scripts rely on pinned local upstream snapshots and inputs; their recorded manifests identify the exact files and hashes. Saved result inspection is possible from this repository, while a fresh end-to-end retrieval run requires the pinned inputs and environment described in [`docs/workspace_context.md`](docs/workspace_context.md). Historical artifacts are not a claim that every experiment has been independently reproduced.

## Start with the saved results

1. Read the [results guide](docs/results_overview.md) and the [hierarchy run report](results/runs/hierarchy_full_20261026_v1/report.md).
2. Inspect [`aggregate_metrics.csv`](results/runs/hierarchy_full_20261026_v1/aggregate_metrics.csv) and [`paired_intervals.csv`](results/runs/hierarchy_full_20261026_v1/paired_intervals.csv) for the exact values and intervals.
3. Read the [V2 report](results/runs/context_full_20261027_v2/report.md) for the heading repair and unchanged retrieval coverage. Its [`paired_intervals.csv`](results/runs/context_full_20261027_v2/paired_intervals.csv) contains the paired comparisons.

The dataset is [IBM watsonxDocsQA](https://huggingface.co/datasets/ibm-research/watsonxDocsQA), pinned to revision `f7da4b6d8a8c4609c49fda06af3bea29b6ef9be4`. The repository is licensed under [MIT](LICENSE).
