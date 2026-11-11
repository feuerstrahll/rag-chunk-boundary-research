# Article review — 27 October 2026

**Needs revision. Review completeness: Partial.** The implemented practical study is coherent as an exploratory investigation of structural packing. The reported principal numerical results agree with the saved experiment. The title, provenance description, comparative framing, literature table and artifact references need corrections. The manuscript and experimental code/results were preserved; this directory contains separate review notes and a diagnostic audit.

Reviewed manuscript: `article/«Разработка и исследование контекстно-зависимой сегментации длинных документов в RAG-системах».docx`. Paragraph identifiers below refer to successive top-level Word paragraphs, including blank paragraphs. Section/table names are the primary reader-facing locations.

**What was independently verified**

- All 39 artifact hashes in the full-run manifest and all 7 hashes in the post-hoc manifest match their files. The 8 preflight source hashes, 11 model-file hashes, corpus hash and frozen gold hash also match.
- All 1,416 saved B1024 contexts were independently scored against the 118 frozen questions and 147 nonoverlapping gold spans using a separate standard-library implementation. All 12 aggregate Coverage@1024 values agree to within 1e-12. Recorded selected-chunk token counts sum to the saved context costs; none exceeds 1,024.
- The four hybrid values are fixed 0.6481433562, structure 0.6942109534, hierarchy 0.7365838348 and unrestricted 0.7281092585.
- Hierarchy minus structure is +4.2373 percentage points: 9 improved questions, 4 worsened questions and 105 ties. Hierarchy minus unrestricted is +0.8475 points: 4 improvements, 3 regressions and 111 ties. A separate 10,000-repetition bootstrap with Python's RNG reproduces the primary interval's interpretation and its rounded endpoints. This is not a bitwise rerun of the original NumPy bootstrap.
- The chunk-count reduction is 20.2597%; indexed tokens increase by 1.5621%. Length-band Coverage values independently reproduce Table 8. The 133 heading records support the reported 121 heading-only / 11 other-body / 1 headings-plus-comments partition, 23 affected documents, and 132 pairs that fit the target and are preserved by the ablation.
- Historical 0.6592 versus 0.8803 and Step 5B figures trace to their upstream summaries. These checks establish consistency with historical records, not independent reproduction of historical runs.

The audit did not rerun model inference, retrieval, tokenizer counts, pytest or an independent full Parquet ranking replay. Top-5 and document-metric values were compared with saved tables and their code definitions. The current default Python lacks the necessary scientific packages; the pinned model files are present and their hashes match. Word pagination, equation rendering and table layout were not visually assessed. Semantic correctness of AI labels was not independently adjudicated.

**Article quality and completeness**

Counts below are scoped inventory counts, not an accuracy percentage or a claim that every component was fully verified. Categories overlap. Zero observed defects does not mean complete verification.

| Category | Observed defects | Assessment |
|---|---|---|
| Usefulness/completeness | 1 / 4 | All four RQs have corresponding analyses. RQ1 has top-5/B1024 comparisons; RQ2 the main contrast; RQ3 an ablation; RQ4 retriever, length and cost results. RQ4's long-document emphasis is weaker than the title suggests; the extra analysis is post hoc. |
| Analytical clarity | 3 / 8 | Inventory: abstract, introduction, Sections 1–4, conclusion, appendix. Introduction/title need narrower framing; Section 2 needs more explicit question-generation/sampling and baseline-objective caveats; appendix needs actual locations. The main statistical caveats are already appropriately stated. |
| Visual/interaction consistency | N/A | No interactive artifact; DOCX visual rendering was not checked. |

**Analytical correctness and robustness**

| Category | Observed defects | Assessment |
|---|---|---|
| Source authority/confidence | 0 / 3 | Historical runs, new run and post-hoc analysis are distinguishable. AI-only/exposed-label limitations are explicit. Independent human semantic validity remains unverified. |
| Value accuracy | 0 / 8 | Six quantitative manuscript tables (3–8) and two historical comparisons were reconciled with saved evidence. Independent raw-context arithmetic covers B1024, length bands and principal effect estimates; other metrics/costs have narrower verification. |
| Within-chart agreement | N/A | No charts in the DOCX. |
| Complete source details | 10 / 11 | Appendix A1–A9 supply descriptions without executable artifact locations; A11 incorrectly says the directory/hashes are unspecified. A10 gives the main run location. |
| Cross-artifact consistency | 1 / 1 | The current root `article/README.md` still presents the early CBO/Stage A plan. It needs a current-status entry pointing to the completed hierarchy study; dated historical notes can remain historical. |
| Data-quality controls | 0 / 4 | Corpus identity, frozen question/span identity, pairing and recorded context-token ceilings passed scoped checks. These do not validate the meaning/completeness of AI evidence. |
| Conclusion support | 1 / 3 | The primary superiority claim is correctly withheld and resource tradeoffs are qualified. The title's broader long-document/context-dependent framing remains stronger than the tested engineering claim. |

**Prioritized problems and proposed fixes**

1. **P1 — Align the title with the actual contribution.** The implementation in `src/article_research/hierarchy_packing.py` uses heading positions, root identity, immediate parents, token limits, overlap and greedy packing. It does not identify semantic dependency, solve a learned/DP boundary objective, or adapt to a question. The manuscript mostly explains this correctly, so the discrepancy is principally positioning. Only 368/1,144 corpus documents (32.17%) and 56/118 evaluated questions (47.46%) belong to the post-hoc >1,024-token category. The hierarchy–ablation mean difference is zero in both length bands above 1,024; the overall +0.85-point difference comes from the ≤510 group. This does not prove that hierarchy never helps long documents.

   **Fix (Proposed):** Prefer «Разработка и исследование иерархической упаковки структурированных технических документов для поиска в RAG-системах». If the current title is retained, explicitly qualify context as structural and foreground the mixed-length corpus and exploratory long-document analysis. Describe the practical contribution as a deterministic packing implementation plus a controlled comparative evaluation with an unconfirmed primary hypothesis. A nonsignificant result does not invalidate the completed engineering work.

2. **P1 — Make the 118-question benchmark provenance reproducible (Section 2.1, P111).** The article discloses AI-only labels and prior exposure, which is correct, but does not fully explain that these are additional generated questions rather than a larger official IBM QA split. The source sampling manifest uses deterministic round-robin allocation across length, heading-depth and structural-complexity strata, seed 2026, from 1,089 eligible documents after excluding 55 exposed documents. Of 120 candidates, `holdout_auto_114` was excluded for insufficient evidence and `holdout_auto_118` for scope. Two AI passes/adjudication replaced the original two-human protocol.

   **Fix (Proposed):** Add a compact description of generation/review, document sampling, exclusions and final counts; distinguish the official 75 QA pairs from the custom 118-question set. Cite `../techdoc-chunking-pipeline/article/results/stage_a/step5a_holdout_preparation_v1/sampling_manifest.json` and `../techdoc-chunking-pipeline/article/results/stage_a/step5a_ai_approved_freeze_v2/{manifest.json,protocol_amendment.md,qc_report.json}`. Retain the current exploratory limitation. A stronger confirmatory claim would require a separately designed independent evaluation, not relabeling this set as a test set.

3. **P1 — Qualify the strength of the fixed baseline for the new endpoint (Section 2.3, P128–129; Section 3.3).** Fixed 480/20 was selected by development EC@5 under a ±2% indexed-token constraint. The new primary endpoint is EC@1024. The selection history is already disclosed, but the article does not explicitly connect that objective mismatch to interpretation of the favorable hierarchy–fixed comparison. The new variant also uses overlap 64 and a hard limit of 510, versus fixed's 20 and 480; the main article correctly reports these differences.

   **Fix (Proposed):** Add: «Fixed 480/20 оптимизирован по EC@5, а не по основному endpoint нового эксперимента EC@1024; поэтому дополнительное сравнение не устанавливает превосходство над fixed baseline, отдельно настроенным для токенного бюджета». Preserve the frozen run. For a future stronger comparison, tune alternatives on development data for the chosen endpoint and evaluate on independent questions; do not retune on the exposed 118.

4. **P1 — Repair the appendix and moderate the reproducibility wording (Appendix A; Section 2.6, P150).** A11 says its independent directory and hashes are unspecified, although `results/analyses/hierarchy_posthoc_20261026_v2/manifest.json` contains them. A1–A9 have no concrete file paths despite P242 promising paths. P150's wording that saved artifacts “ensure reproducibility” exceeds what artifact replay alone demonstrates.

   **Fix (Proposed):** Give A11 the directory above and its manifest; add concrete versioned paths for A1–A9. Replace P150's opening with wording that the saved artifacts support verification and rerunning, while distinguishing the completed original run, metric replay, and a separate independent end-to-end rerun. Remove “по сообщённой проверке” in favor of an explicit validation artifact. Keep 34 passed as a historical test record unless tests are actually rerun.

   Useful source locations, relative to `article/`:

   - A1: `../techdoc-chunking-pipeline/results/watsonx_budget_matched/summary.json`.
   - A2: `../techdoc-chunking-pipeline/article/results/stage_a/step3_20261018_v2/`.
   - A3: `../techdoc-chunking-pipeline/article/results/stage_a/step5b_mechanism_diagnostics_20261020_v1/` (budget/context diagnostics; cite the exact consumed table).
   - A4: `../techdoc-chunking-pipeline/article/results/stage_a/step4_20261018_v2/baseline_selection.json`.
   - A5: `../techdoc-chunking-pipeline/article/results/stage_a/step5b_20261020_v1/summary.json`.
   - A6: `../techdoc-chunking-pipeline/article/results/stage_a/step5b_exploratory_diagnostics_20261020_v1/`.
   - A7: `../techdoc-chunking-pipeline/article/results/stage_a/step5b_mechanism_interpretation_20261020_v1/diagnostic_synthesis.md` and its underlying mechanism diagnostics.
   - A8: `../techdoc-chunking-pipeline/article/results/stage_a/step5b_index_coverage_diagnostic_20261020_v1/`.
   - A9: `results/runs/stage_a_train_run1_20261015/execution_cost.json` (identify the chosen repeat explicitly).

5. **P2 — Correct the review table and missing citations (Table 1 and introduction).** Table 1's combined semantic/model-based row describes all cited methods [4,6,21] as embedding-similarity boundary methods. This contradicts the more accurate Section 1.3: TextTiling is lexical; Meta-Chunking uses language-model uncertainty/perplexity and margin signals, plus dynamic merging and semantic completion. Its latter components also act beyond physical boundary selection. [Meta-Chunking primary paper](https://arxiv.org/html/2410.12788v3), [TextTiling primary publication](https://aclanthology.org/J97-1003/).

   **Fix (Proposed):** Separate lexical/embedding-coherence methods from uncertainty-driven LLM methods, or broaden the mechanism and stage columns to accurately cover each. Add the missing bibliography entry for FreeChunker, mentioned in P022: Zhang W., Jiang Y.-H., Cao Y., Wu Y., *FreeChunker: A Cross-Granularity Chunking Framework*, arXiv:2510.20356, citing v2 if that is the reviewed version. [Primary source](https://arxiv.org/abs/2510.20356). Add in-text citations [1] and [2] to the RAG and Lost-in-the-Middle introduction claims; they are listed but not cited in the body. Add an explicit dataset citation for the IBM source; the official 1,144/75/25-human/50-synthetic description is supported by its [dataset card](https://huggingface.co/datasets/ibm-research/watsonxDocsQA).

6. **P2 — Strengthen the practical explanation without rewriting historical outcomes.** The 132 parent-heading/body separations are a real limitation of `_regions`: entering a child section changes the immediate-parent region before `_units` can bind the ancestor heading to that child's text. The manuscript correctly avoids calling these 132 retrieval failures; only one evaluated question is from affected documents.

   **Fix (Proposed):** Add concise pseudocode or a worked H1/H2/H3 example showing the region key, target/hard-limit distinction, overlap and fallback. Add the independent wins/losses/ties above to make effect size concrete. A future algorithm revision can bind a parent-heading chain to initial descendant content when permitted by the budget, but must get a new version and new evaluation; the current run should stay intact.

7. **P2 — Synchronize the project entry point and candidate-pool caveat.** `README.md` and several early documents describe a planned CBO and unfinished Stage A, whereas the hierarchy protocol records the later authorized exploratory study. Preserve dated historical notes, but add a clear current-status navigation entry to the completed run, protocol and results. The recorded B1024 packing scans all available BM25/dense candidates, whereas RRF uses the union of depth-100 component lists. Selected chunk ranks reach 5,131 for BM25 and 178 for hybrid. This is consistent with the written skip-nonfitting policy, not an arithmetic defect.

   **Fix (Proposed):** State that cross-retriever differences describe complete configured pipelines, including their candidate pools. Do not attribute the difference solely to the scoring function. A fixed candidate-depth sensitivity analysis would be a separate exploratory analysis, not a silent alteration of the frozen experiment.

**Assessment of the practical concept**

The progression from Stage A diagnostics, through an unsuccessful dependency pilot, to an explicitly structural engineering hypothesis is defensible. The paired pilot reviewed seven positive dependency cases: six were preserved by both existing methods, and one only by structure-aware; no confirmed structure-aware break justified a new dependency cost or DP optimizer. The later hierarchy experiment therefore changes the research question rather than proving the original dependency-optimization idea. The article should present that narrower question consistently.

The manuscript already gets several important points right: it separates retrieval from generation, defines character coverage and threshold-based success, discloses exposed AI labels, distinguishes the historical fixed baseline, uses paired comparisons, reports residual token costs and withholds claims of hierarchy superiority or stable speedup. The study can stand as an exploratory implementation-and-evaluation article after the corrections above. Strong claims about semantic dependencies, long-document superiority, independent confirmation or improved generated answers would require additional evidence.

Reproducible supporting calculations: [checks.json](checks.json) and [verify_saved_results.py](verify_saved_results.py). These are new diagnostic review artifacts, not new retrieval experiments.
