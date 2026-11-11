"""Evaluate a completed frozen V2 run; no encoder inference or new retrieval."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

from run_context_packing import check_hashes, write_jsonl
from run_hierarchy_packing import (
    ARTICLE,
    digest,
    load_documents,
    now,
    read_json,
    resolve_inputs,
    validate_chunks,
    write_json,
)


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def groups_for_evaluation(documents, config, development):
    from article_research.context_evaluation import enumerate_heading_links

    heading_links, groups = [], []
    for document in documents:
        links = enumerate_heading_links(document)
        heading_links.extend({"doc_id": document.doc_id, **r} for r in links)
        by_body = defaultdict(list)
        for link in links:
            by_body[link["b_block_id"]].append(link)
        for block_id, chain in by_body.items():
            groups.append({"group_id": f"chain:{block_id}", "doc_id": document.doc_id,
                           "cohort": "structural_chain", "b_span": chain[0]["b_span"],
                           "a_spans": [r["a_spans"][0] for r in chain],
                           "label_provenance": "structural_definition_not_semantic_annotation",
                           "human_adjudicated": False})
    labels = [json.loads(line) for line in (ARTICLE / config["semantic_labels"]).read_text().splitlines()]
    reviewed = [r for r in labels if r["decision"] == "positive" and r["review_status"] == "content_checked"]
    proposals = [r for r in development["examples"] if r["label_provenance"] == "new_development_AI_proposal"]
    for cohort, records in [("existing_content_checked_AI", reviewed), ("new_development_AI_proposals", proposals)]:
        by_id = defaultdict(list)
        for row in records:
            by_id[(row["doc_id"], row["record_id"])].append(row)
        for (doc_id, record_id), edges in by_id.items():
            assert all(r["b_span"] == edges[0]["b_span"] for r in edges)
            groups.append({"group_id": record_id, "doc_id": doc_id, "cohort": cohort,
                           "b_span": edges[0]["b_span"], "a_spans": [r["a_span"] for r in edges],
                           "label_provenance": "AI_ChatGPT_content_checked" if records is reviewed else "AI_Codex_proposal",
                           "human_adjudicated": False})
    docs = {d.doc_id: d for d in documents}
    for row in [*reviewed, *proposals]:
        text = docs[row["doc_id"]].canonical_text
        for key in ["a", "b"]:
            start, end = row[key + "_span"]
            if text[start:end] != row[key + "_quote"]:
                raise ValueError("Semantic label canonical quote mismatch")
    return heading_links, groups


def independent_metric_check(folder, chunks, gold, reported):
    """Recompute all endpoints from saved rankings using independent union math."""
    import pyarrow.parquet as pq

    questions = {q["question_id"]: q for q in gold}
    metrics = {(r["query_id"], r["method"]): r for r in reported}
    contexts = {(r["query_id"], r["method"]): r for r in
                (json.loads(line) for line in (folder / "contexts.jsonl").read_text().splitlines())}
    top_contexts, checked, seen = {}, 0, set()
    file = pq.ParquetFile(folder / "rankings.parquet")
    for number in range(file.num_row_groups):
        rows = file.read_row_group(number).to_pylist()
        query_id, method = rows[0]["query_id"], rows[0]["method"]
        key = query_id, method
        if key in seen or any((r["query_id"], r["method"]) != key for r in rows):
            raise ValueError("Ranking row groups are not complete unique query/method lists")
        seen.add(key)
        ranked = [chunks[r["chunk_index"]] for r in rows]
        keys = [(-r["score"], c["doc_id"], c["chunk_id"]) for r, c in zip(rows, ranked)]
        if keys != sorted(keys) or [r["rank"] for r in rows] != list(range(1, len(rows) + 1)):
            raise ValueError("Ranking/tie ordering changed")
        question = questions[query_id]
        spans = [(r["char_start"], r["char_end"]) for r in question["gold_spans"]]

        def coverage(selected, spans=spans, question=question):
            total = 0
            for left, right in spans:
                cursor = left
                for a, b in sorted((max(left, c["char_start"]), min(right, c["char_end"]))
                                   for c in selected if c["doc_id"] == question["doc_id"]):
                    a = max(a, cursor)
                    if b > a:
                        total += b - a
                        cursor = b
            return total / sum(b - a for a, b in spans)

        used, selected = 0, []
        for c in ranked:
            if used + c["token_count"] <= 1024:
                selected.append(c)
                used += c["token_count"]
        saved = contexts[key]
        if used != saved["used_tokens"] or [c["chunk_id"] for c in selected] != [c["chunk_id"] for c in saved["chunks"]]:
            raise ValueError("Saved context differs from original greedy whole-chunk packing")
        top = ranked[:5]
        top_contexts[key] = top
        doc_order = list(dict.fromkeys(c["doc_id"] for c in ranked))
        doc_rank = doc_order.index(question["doc_id"]) + 1 if question["doc_id"] in doc_order else None
        evidence_rank = next((i for i, c in enumerate(top, 1) if coverage([c]) >= .8), None)
        expected = {
            "EvidenceCoverage@5": coverage(top), "EvidenceRecall@5": float(coverage(top) >= .8),
            "EvidenceMRR@5": 1 / evidence_rank if evidence_rank else 0.,
            "EvidenceCoverage@B=1024": coverage(selected), "actual_context_tokens_B1024": used,
            "DocumentRecall@5": float(doc_rank is not None and doc_rank <= 5),
            "DocumentMRR@10": 1 / doc_rank if doc_rank is not None and doc_rank <= 10 else 0.,
        }
        for name, value in expected.items():
            if not math.isclose(value, metrics[key][name], abs_tol=1e-12):
                raise ValueError(f"Independent metric mismatch: {folder.name}/{key}/{name}")
        checked += 1
    if seen != metrics.keys() or seen != contexts.keys():
        raise ValueError("Incomplete metric/context population")
    return contexts, top_contexts, {"query_method_rows": checked, "all_endpoints_recomputed": True,
                                  "saved_budget_contexts_verified": True, "ranking_ties_verified": True}


def binding_evaluation(groups, chunks_by_variant, contexts, top_contexts, gold, tokenizer, docs):
    from article_research.context_evaluation import index_preserved, retrieved_state

    by_doc = defaultdict(list)
    for group in groups:
        by_doc[group["doc_id"]].append(group)
    counts = Counter(g["cohort"] for g in groups)
    index_rows, retrieval_rows, summaries = [], [], []
    gold_docs = {q["question_id"]: q["doc_id"] for q in gold}
    gold_pairs = Counter(g["cohort"] for q in gold for g in by_doc[q["doc_id"]])
    for variant, chunks in chunks_by_variant.items():
        local = defaultdict(list)
        for c in chunks:
            local[c["doc_id"]].append(c)
        for group in groups:
            left = min(a for a, _ in group["a_spans"] + [group["b_span"]])
            right = max(b for _, b in group["a_spans"] + [group["b_span"]])
            tokens = len(tokenizer(docs[group["doc_id"]].canonical_text[left:right],
                                   add_special_tokens=False, truncation=False)["input_ids"])
            index_rows.append({"variant": variant, "group_id": group["group_id"], "doc_id": group["doc_id"],
                               "cohort": group["cohort"], "preserved": index_preserved(group, local[group["doc_id"]]),
                               "continuous_interval_tokens": tokens, "fits_target": tokens <= 480,
                               "fits_encoder": tokens <= 510, "label_provenance": group["label_provenance"],
                               "human_adjudicated": False})
        for mode in ["B1024", "top5"]:
            accumulated = defaultdict(Counter)
            for (query_id, method), saved in contexts[variant].items():
                selected = saved["chunks"] if mode == "B1024" else top_contexts[variant][(query_id, method)]
                for doc_id in sorted({c["doc_id"] for c in selected}):
                    for group in by_doc[doc_id]:
                        state = retrieved_state(group, selected)
                        retrieval_rows.append({"variant": variant, "query_id": query_id, "method": method,
                                               "context": mode, "group_id": group["group_id"], "doc_id": doc_id,
                                               "cohort": group["cohort"], **state,
                                               "is_gold_document": doc_id == gold_docs[query_id],
                                               "label_provenance": group["label_provenance"]})
                        for scope in ["all_indexed_documents", "gold_document_only"]:
                            if scope == "gold_document_only" and doc_id != gold_docs[query_id]:
                                continue
                            entry = accumulated[(method, group["cohort"], scope)]
                            entry["touched_query_group_pairs"] += 1
                            entry["b_fully_returned"] += state["b_fully_returned"]
                            entry["complete"] += state["complete"]
            for method in ["bm25", "dense", "hybrid"]:
                for cohort, group_count in counts.items():
                    for scope in ["all_indexed_documents", "gold_document_only"]:
                        entry = accumulated[(method, cohort, scope)]
                        denominator = group_count * len(gold) if scope == "all_indexed_documents" else gold_pairs[cohort]
                        b, complete = entry["b_fully_returned"], entry["complete"]
                        summaries.append({"variant": variant, "method": method, "context": mode,
                                          "cohort": cohort, "scope": scope, "groups": group_count,
                                          "eligible_query_group_pairs": denominator,
                                          "touched_query_group_pairs": entry["touched_query_group_pairs"],
                                          "b_fully_returned": b, "complete": complete,
                                          "b_returned_missing_a": b - complete,
                                          "complete_fraction_all_pairs": complete / denominator if denominator else None,
                                          "complete_given_b_returned": complete / b if b else None,
                                          "human_adjudicated_groups": 0})
    return index_rows, retrieval_rows, summaries


def analyze(output):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from article_research.context_evaluation import index_preserved
    from article_research.packing_evaluation import (
        METRICS,
        document_clusters,
        paired_cluster_bootstrap,
    )
    from transformers import AutoTokenizer

    if (output / "manifest.json").exists() or (output / "summary.json").exists():
        raise FileExistsError("Never overwrite completed analysis")
    retrieval_manifest = read_json(output / "retrieval_manifest.json")
    if retrieval_manifest["status"] != "completed_new_exploratory_retrieval":
        raise ValueError("A full completed V2 retrieval run is required")
    immutable = {output / name: sha for name, sha in retrieval_manifest["artifact_sha256"].items()}
    check_hashes(immutable)
    config, preflight = read_json(output / "frozen_config.json"), read_json(output / "preflight.json")
    check_hashes(preflight["source_hashes"])
    check_hashes(preflight["immutable_source_artifacts"])
    _, _, _, model = resolve_inputs(config)
    check_hashes({model / name: sha for name, sha in preflight["model_hashes"].items()})
    tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True, use_fast=True)
    documents, _, gold, _ = load_documents(config)
    docs = {d.doc_id: d for d in documents}
    heading_links, groups = groups_for_evaluation(documents, config, read_json(output / "development_examples.json"))
    cohort_counts = Counter(g["cohort"] for g in groups)
    if cohort_counts["existing_content_checked_AI"] != 7 or cohort_counts["new_development_AI_proposals"] != 2:
        raise ValueError("Frozen AI-label evaluation cohorts changed")
    baseline = Path(preflight["baseline_run"])
    folders = {"v1": baseline / "hierarchy", **{v: output / v for v in config["variants"]}}
    chunks, contexts, tops, metrics, validation, costs = {}, {}, {}, [], {}, {}
    for variant, folder in folders.items():
        chunks[variant] = pq.read_table(folder / "chunks.parquet").to_pylist()
        local = defaultdict(list)
        for c in chunks[variant]:
            local[c["doc_id"]].append(SimpleNamespace(**c))
        integrity = [validate_chunks(d, local[d.doc_id], tokenizer) for d in documents]
        rows = pq.read_table(folder / "per_question_metrics.parquet").to_pylist()
        contexts[variant], tops[variant], validation[variant] = independent_metric_check(folder, chunks[variant], gold, rows)
        validation[variant].update({"all_chunks_literal_token_counts_and_coverage_checked": True,
                                    "uncovered_nonwhitespace_characters": sum(r["uncovered_nonwhitespace_characters"] for r in integrity)})
        metrics.extend({**row, "variant": variant} for row in rows)
        costs[variant] = read_json(folder / "index_statistics.json")
        if max(c["token_count"] for c in chunks[variant]) > costs[variant]["encoder_text_budget"]:
            raise ValueError("Saved chunks exceed the resolved encoder budget")
        if (len(chunks[variant]) != costs[variant]["chunk_count"]
                or sum(c["token_count"] for c in chunks[variant]) != costs[variant]["indexed_tokens"]):
            raise ValueError("Recorded index costs differ from actual chunks")
        costs[variant]["derived_local_encode_calls"] = 1 + len(gold)
        costs[variant]["derived_document_forward_batches"] = math.ceil(len(chunks[variant]) / config["model"]["batch_size"])
        costs[variant]["derived_query_forward_batches"] = len(gold)
        costs[variant]["derived_actual_overlap_fraction_of_index_tokens"] = (
            costs[variant]["actual_overlap_tokens_sum"] / costs[variant]["indexed_tokens"])
        costs[variant]["measurement_origin"] = "reused_completed_v1" if variant == "v1" else "new_isolated_process"
        print(f"validated {variant}: {len(chunks[variant])} chunks; {len(rows)} metric rows", flush=True)
    expected = {(v, m, q["question_id"]) for v in folders for m in config["retrieval"]["methods"] for q in gold}
    assert len(metrics) == len(expected) == 1062
    assert {(r["variant"], r["method"], r["query_id"]) for r in metrics} == expected
    aggregates = []
    for variant in folders:
        for method in config["retrieval"]["methods"]:
            rows = [r for r in metrics if r["variant"] == variant and r["method"] == method]
            aggregates.append({"variant": variant, "method": method, "questions": len(rows),
                               **{name: float(np.mean([r[name] for r in rows])) for name in METRICS},
                               "mean_actual_context_tokens": float(np.mean([r["actual_context_tokens_B1024"] for r in rows]))})
    clusters = document_clusters({q["question_id"]: [q["doc_id"]] for q in gold})
    intervals = []
    for reference, candidate in [("v1", "headings"), ("headings", "headings_intro"), ("v1", "headings_intro")]:
        for method in config["retrieval"]["methods"]:
            for name in METRICS:
                def values(variant, name=name, method=method):
                    return {r["query_id"]: r[name] for r in metrics if r["variant"] == variant and r["method"] == method}
                intervals.append({"contrast": f"{candidate}_minus_{reference}", "method": method, "metric": name,
                                  **paired_cluster_bootstrap(values(reference), values(candidate), clusters,
                                      repetitions=config["evaluation"]["bootstrap_repetitions"],
                                      seed=config["evaluation"]["bootstrap_seed"])})
    local_chunks = {v: defaultdict(list) for v in folders}
    for variant, rows in chunks.items():
        for row in rows:
            local_chunks[variant][row["doc_id"]].append(row)
    from article_research.context_packing import (
        ContextAwarePackingChunker,
        ContextPackingConfig,
    )
    from chunking.fixed import TokenizerConfig

    baseline_parameters = read_json(baseline / "frozen_config.json")["variants"]["hierarchy"]
    off = ContextAwarePackingChunker(ContextPackingConfig(
        tokenizer=TokenizerConfig(name=config["model"]["name"], revision=config["model"]["revision"]),
        max_tokens=preflight["baseline_encoder_text_budget"], **baseline_parameters,
        heading_chains=False, intro_rule=False), tokenizer=tokenizer)
    fields = ["chunk_id", "char_start", "char_end", "text", "token_count", "fallback_split"]
    for document in documents:
        actual = off.chunk(document)
        saved = local_chunks["v1"][document.doc_id]
        if [[getattr(c, k) for k in fields] for c in actual] != [[c[k] for k in fields] for c in saved]:
            raise ValueError("Both flags off do not reproduce the saved V1 segmentation")
    v1_equivalence = {"all_documents": len(documents), "all_chunks": len(chunks["v1"]),
                      "both_flags_off_match_v1_spans_ids_text_tokens_fallback": True,
                      "encoder_inference": False}
    write_json(output / "v1_equivalence.json", v1_equivalence)
    known = [json.loads(line) for line in (ARTICLE / config["heading_audit"] / "heading_cases.jsonl").read_text().splitlines()]
    known_ids = {(r["doc_id"], r["diagnostic_block_id"]) for r in known if r["first_subtree_body"]}
    assert len(known_ids) == 132
    expected_links = {(r["doc_id"], r["link_id"]): r for r in heading_links}
    for case in known:
        text = docs[case["doc_id"]].canonical_text
        for span in [*case["heading_run"], *([case["first_subtree_body"]] if case["first_subtree_body"] else [])]:
            if text[span["char_start"]:span["char_end"]] != span["text"]:
                raise ValueError("Known heading audit canonical quote mismatch")
        key = case["doc_id"], case["diagnostic_block_id"]
        if case["first_subtree_body"]:
            body = case["first_subtree_body"]
            if expected_links[key]["b_span"] != [body["char_start"], body["char_end"]]:
                raise ValueError("Independent subtree scan disagrees with old 132 cases")
        elif key in expected_links:
            raise ValueError("An empty root was assigned another root's content")
    for variant in config["variants"]:
        detected = [json.loads(line) for line in (output / variant / "preferred_links.jsonl").read_text().splitlines()]
        actual = {(r["doc_id"], r["link_id"]): r for r in detected if r["kind"] == "heading"}
        if actual.keys() != expected_links.keys() or any(actual[k]["b_span"] != r["b_span"] for k, r in expected_links.items()):
            raise ValueError("Independent heading enumeration differs from the frozen detector")
    heading_rows = []
    for link in heading_links:
        text = docs[link["doc_id"]].canonical_text
        tokens = len(tokenizer(text[link["start"]:link["end"]], add_special_tokens=False, truncation=False)["input_ids"])
        heading_rows.append({**link, "tokens": tokens, "fits_target": tokens <= 480, "fits_encoder": tokens <= 510,
                             "known_132": (link["doc_id"], link["link_id"]) in known_ids,
                             **{v: index_preserved(link, local_chunks[v][link["doc_id"]]) for v in folders}})
    heading_summary = {}
    for variant in folders:
        heading_summary[variant] = {
            "all_heading_body_links": len(heading_rows), "known_cases": 132,
            "known_cases_preserved": sum(r["known_132"] and r[variant] for r in heading_rows),
            "all_preserved": sum(r[variant] for r in heading_rows),
            "broken_fits_target": sum(not r[variant] and r["fits_target"] for r in heading_rows),
            "broken_fits_encoder": sum(not r[variant] and r["fits_encoder"] for r in heading_rows),
            "broken_exceeds_encoder": sum(not r[variant] and not r["fits_encoder"] for r in heading_rows),
            "new_broken_relative_to_v1": sum(r["v1"] and not r[variant] for r in heading_rows),
            "new_feasible_broken_relative_to_v1": sum(r["v1"] and not r[variant] and r["fits_encoder"] for r in heading_rows),
        }
    index_rows, retrieval_rows, completeness = binding_evaluation(groups, chunks, contexts, tops, gold, tokenizer, docs)
    index_summary = []
    for variant in folders:
        for cohort in sorted({g["cohort"] for g in groups}):
            rows = [r for r in index_rows if r["variant"] == variant and r["cohort"] == cohort]
            index_summary.append({"variant": variant, "cohort": cohort, "groups": len(rows),
                                  "preserved": sum(r["preserved"] for r in rows),
                                  "broken_fits_encoder": sum(not r["preserved"] and r["fits_encoder"] for r in rows),
                                  "broken_exceeds_encoder": sum(not r["preserved"] and not r["fits_encoder"] for r in rows),
                                  "human_adjudicated_groups": 0})
    detector = {}
    for variant in config["variants"]:
        links = [json.loads(line) for line in (output / variant / "preferred_links.jsonl").read_text().splitlines()]
        intro = [r for r in links if r["kind"] == "intro"]
        candidates = [json.loads(line) for line in (output / variant / "intro_candidates.jsonl").read_text().splitlines()]
        detector[variant] = {"intro_links_proposed": len(intro), "documents_with_intro_links": len({r["doc_id"] for r in intro}),
                             "intro_links_preserved": sum(r["preserved_in_index"] for r in intro),
                             "intro_links_exceed_encoder": sum(not r["fits_encoder"] for r in intro),
                             "packing_decisions": dict(Counter(r["packing_decision"] for r in intro)),
                             "candidate_decisions": dict(Counter(r["decision"] for r in candidates)),
                             "candidate_units": {"intro_only_without_following_list": sum("b_block_id" not in r for r in candidates),
                                                 "intro_item_pairs": sum("b_block_id" in r for r in candidates)},
                             "abstention_reasons": dict(Counter(r["reason"] for r in candidates if r["decision"] == "abstain")),
                             "label_provenance": "AI_proposed_heuristic_not_semantic_ground_truth"}
    equality = []
    for reference, candidate in [("v1", "headings"), ("headings", "headings_intro")]:
        same = sum([(c["char_start"], c["char_end"]) for c in local_chunks[reference][d.doc_id]] ==
                   [(c["char_start"], c["char_end"]) for c in local_chunks[candidate][d.doc_id]] for d in documents)
        same_ids = sum([c["chunk_id"] for c in local_chunks[reference][d.doc_id]] ==
                       [c["chunk_id"] for c in local_chunks[candidate][d.doc_id]] for d in documents)
        if same_ids != same:
            raise ValueError("Identical geometry must preserve V1-compatible chunk identities")
        equality.append({"reference": reference, "candidate": candidate, "documents": len(documents),
                         "identical_spans": same, "identical_spans_and_ids": same_ids})
    labeled_docs = {g["doc_id"] for g in groups if g["cohort"] != "structural_chain"}
    summary = {
        "status": "completed_exploratory_continuation_exposed_AI_labels", "questions": len(gold),
        "documents": len(documents), "cluster_count": len(set(clusters.values())), "paired_metric_rows": len(metrics),
        "aggregates": aggregates, "paired_intervals": intervals, "costs": costs,
        "heading_behavior": heading_summary, "index_preservation": index_summary,
        "known_empty_root_examples": [{"doc_id": r["doc_id"], "heading_run": r["heading_run"],
                                       "next_heading": r["next_heading"]} for r in known if r["first_subtree_body"] is None],
        "retrieved_completeness": completeness, "detector": detector, "span_equality": equality,
        "semantic_label_status": {"human_adjudicated_groups": 0,
                                  "existing_AI_groups": cohort_counts["existing_content_checked_AI"],
                                  "new_AI_proposal_groups": cohort_counts["new_development_AI_proposals"],
                                  "labeled_documents": len(labeled_docs),
                                  "evaluation_questions_in_labeled_documents": sum(q["doc_id"] in labeled_docs for q in gold),
                                  "conclusion_scope": "mechanical_AI_label_retention_only_no_human_validated_semantic_effect"},
        "validation": validation, "v1_equivalence": v1_equivalence,
        "software_checks": read_json(output / "test_verification.json") if (output / "test_verification.json").exists() else None,
    }
    pq.write_table(pa.Table.from_pylist(metrics), output / "per_question_metrics.parquet", compression="zstd")
    write_csv(output / "aggregate_metrics.csv", aggregates)
    write_csv(output / "paired_intervals.csv", intervals)
    write_csv(output / "index_preservation.csv", index_rows)
    write_csv(output / "retrieved_completeness.csv", completeness)
    write_jsonl(output / "retrieved_group_details.jsonl", retrieval_rows)
    write_jsonl(output / "heading_behavior.jsonl", heading_rows)
    write_jsonl(output / "evaluation_groups.jsonl", groups)
    write_json(output / "summary.json", summary)
    write_json(output / "validation.json", {"status": "passed_for_retrieval_artifacts", "variants": validation,
               "all_1062_metric_rows_independently_recomputed": True, "old_sources_and_artifacts_unchanged": True,
               "software_checks": summary["software_checks"]})
    render(output, summary, heading_rows, groups, docs, local_chunks)
    check_hashes(immutable)
    check_hashes(preflight["source_hashes"])
    check_hashes(preflight["immutable_source_artifacts"])
    analysis_files = [Path(__file__).resolve(), ARTICLE / "src/article_research/context_evaluation.py",
                      ARTICLE / "scripts/context_packing_report.py", ARTICLE / "tests/test_context_evaluation.py",
                      ARTICLE / "tests/test_context_analysis.py", ARTICLE / "tests/test_context_reporting.py"]
    write_json(output / "manifest.json", {
        "status": summary["status"], "finished_at_utc": now(),
        "retrieval_manifest_sha256": digest(output / "retrieval_manifest.json"),
        "analysis_source_hashes": {str(p): digest(p) for p in analysis_files},
        "analysis_protocol": "pre-retrieval context_packing_v2_protocol.md",
        "analysis_sources_finalized_after_retrieval_started": True,
        "old_artifacts_unchanged": True, "encoder_inference_during_analysis": False,
        "current_git_status": "unavailable_git_executable_absent; executed_code_identified_by_SHA256",
        "preflight_git_metadata_origin": "reference_v1_run_not_a_fresh_git_inspection",
        "artifact_sha256": {str(p.relative_to(output)): digest(p) for p in sorted(output.rglob("*")) if p.is_file()},
    })
    print(json.dumps({"status": summary["status"], "report": str(output / "report.md")}), flush=True)
    return summary


def render(output, summary, heading_rows, groups, docs, local_chunks):
    """Implemented below; final narrative uses the frozen endpoint, never tunes it."""
    from context_packing_report import render_report

    render_report(output, summary, heading_rows, groups, docs, local_chunks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    output = args.run.resolve()
    if not output.is_relative_to(ARTICLE / "results/runs"):
        raise ValueError("Analysis must stay in article/results/runs")
    analyze(output)


if __name__ == "__main__":
    main()
