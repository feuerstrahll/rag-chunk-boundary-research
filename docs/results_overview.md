# Results guide

## Research question and evidence

The comparison asks whether changing chunk boundaries improves retrieval of aligned gold-evidence spans from technical documents under fixed parsing, embedding, and retrieval settings. The corpus contains 1,144 IBM watsonxDocsQA documents in Markdown form. The main article comparisons use 118 additional AI-generated questions from previously unused documents. These are distinct from the dataset's original 75 questions.

The 118-question labels received AI-only review. Their completeness and semantic correctness have not been independently established by human annotators. The saved experiments therefore support findings **for this evaluated set and its labels**, not a general claim about answer quality or other corpora.

## How to read the metrics

- **EvidenceCoverage@5:** fraction of aligned gold-evidence characters covered by the global top five chunks.
- **EvidenceCoverage at B=1024:** the same coverage after packing whole chunks in rank order under a 1,024-token context limit; a chunk that does not fit is skipped rather than truncated.
- Coverage is averaged across questions. The 95% intervals use paired bootstrap sampling by gold-document cluster, 10,000 repetitions, seed 2026. The hierarchy and later analyses are exploratory; their intervals are descriptive and unadjusted for multiple comparisons.

Top five chunks can contain very different numbers of tokens across methods. That is why top-5 and token-limited coverage answer different questions. Neither metric measures generated-answer correctness or semantic sufficiency.

## Hybrid retrieval results

| Method | EC@5 | EC@1024 | Indexed chunks | Indexed tokens |
| --- | ---: | ---: | ---: | ---: |
| Fixed-size 480/20 | 0.8515 | 0.6481 | 3,316 | 1,311,053 |
| Structure-aware | 0.7797 | 0.6942 | 6,007 | 1,316,605 |
| Hierarchical packing V1 | 0.8390 | 0.7366 | 4,790 | 1,337,172 |
| Unrestricted packing ablation | 0.8729 | 0.7281 | 3,859 | 1,356,371 |
| Hierarchical packing V2 | 0.8390 | 0.7366 | 4,661 | 1,337,363 |

These rows are drawn from the [V1 run report](../results/runs/hierarchy_full_20261026_v1/report.md) and [V2 run report](../results/runs/context_full_20261027_v2/report.md). They describe measured outcomes, not a matched-token-index causal estimate. BM25 and dense retrieval results are in those reports.

### Fixed-size versus structure-aware

The frozen fixed-size configuration was selected from a development grid for EC@5. On the 118-question set, fixed-size scored 0.8515 versus 0.7797 for structure-aware at top-5. The paired difference *structure minus fixed* was −0.0719, 95% interval [−0.1437, −0.0041]. At B=1024, the scores were 0.6481 and 0.6942, respectively; the difference was +0.0461, 95% interval [−0.0363, +0.1305]. The sign reverses, while the B=1024 interval includes zero. [Source and protocol note](results_section_draft.md).

### Hierarchical packing

At B=1024, V1 scored 0.7366 versus 0.6942 for structure-aware. The paired difference was +0.0424, descriptive 95% interval [−0.0169, +0.1017]. Against unrestricted packing, V1 differed by +0.0085, interval [−0.0339, +0.0508]. The specific benefit of the hierarchy constraint has not been established. The [run report](../results/runs/hierarchy_full_20261026_v1/report.md) and [paired intervals](../results/runs/hierarchy_full_20261026_v1/paired_intervals.csv) contain all retriever variants and contrasts.

### Heading repair in V2

V1 separated 132 known heading-to-first-body links that could fit within the encoder budget. V2 repaired all 132 and preserved all 4,339 evaluated heading–body links, including all 4,207 complete heading chains. The number of chunks fell from 4,790 to 4,661; indexed tokens rose from 1,337,172 to 1,337,363. For all 118 questions, EC@5 and EC@1024 were exactly unchanged for BM25, dense, and hybrid retrieval. Structural preservation and retrieval effectiveness are distinct outcomes. See the [V2 report](../results/runs/context_full_20261027_v2/report.md) and [independent saved-artifact recheck](../results/reviews/article_recheck_20261027_v2/review.md).

## Diagnostic interpretation

Fixed-size and structure-aware differed at top-5 on 19 of 118 questions: fixed-size led on 14, structure-aware on 5, and the remaining 99 tied. In 18 of the 19 non-ties, the losing method's gold-bearing chunk appeared below rank five in the fused ranking. This post hoc finding points to ranking as a frequent immediate cause of top-5 misses; it does not isolate a causal effect of a particular boundary. [Diagnostic source](results_section_draft.md).

## Reproduction boundary

The saved metrics and key intervals were checked against stored chunks, rankings, and spans in the [article recheck](../results/reviews/article_recheck_20261027_v2/review.md). That check did not rerun embeddings or the full retrieval pipeline. Some historical upstream manifests originated from dirty source trees; the exact historical code state is not fully reconstructible from those manifests alone. A full independent rerun, human gold review, and generated-answer evaluation remain separate work. The [workspace context](workspace_context.md) records the pinned corpus and model revisions and the local input requirements.
