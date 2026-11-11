"""Offline V2 experiment, using the unchanged V1 retrieval/scoring loop.

New outputs only. Reuse V1 only after complete protocol/provenance validation.
The worker derives from run_hierarchy_packing.worker, changing only segmentation
construction and diagnostic sidecars; retrieval, encoder and packing are intact.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from run_hierarchy_packing import (
    ARTICLE,
    RssSampler,
    digest,
    load_documents,
    now,
    read_json,
    resolve_inputs,
    validate_chunks,
    write_json,
)


def write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def verified_artifacts(folder):
    manifest = read_json(folder / "manifest.json")
    expected = {folder / name: sha for name, sha in manifest["artifact_sha256"].items()}
    expected[folder / "manifest.json"] = digest(folder / "manifest.json")
    check_hashes(expected)
    return {str(p): sha for p, sha in expected.items()}


def check_hashes(expected):
    for path, sha in expected.items():
        if digest(path) != sha:
            raise ValueError(f"Immutable artifact changed: {path}")


def verify_baseline(config):
    """Fail closed on any protocol, artifact, code, input or environment mismatch."""
    from run_stage_a_baseline import environment

    baseline = (ARTICLE / config["baseline_run"]).resolve()
    old = read_json(baseline / "frozen_config.json")
    preflight = read_json(baseline / "preflight.json")
    immutable = verified_artifacts(baseline)
    if read_json(baseline / "manifest.json")["status"] != "completed_exploratory_exposed_ai_labels":
        raise ValueError("V1 is not a completed full experiment")
    for key in ["dataset", "gold", "snapshot", "model", "retrieval", "cost", "seed"]:
        if old[key] != config[key]:
            raise ValueError(f"V1 protocol differs: {key}")
    for key, value in old["evaluation"].items():
        if key != "primary_contrast" and value != config["evaluation"][key]:
            raise ValueError(f"V1 metric/budget protocol differs: {key}")
    for parameters in config["variants"].values():
        if {k: v for k, v in parameters.items() if k not in {"heading_chains", "intro_rule"}} != old["variants"]["hierarchy"]:
            raise ValueError("V2 changes non-component packing parameters")
    if config["variants"]["headings_intro"] != {**config["variants"]["headings"], "intro_rule": True}:
        raise ValueError("Intro ablation changes other parameters")
    snapshot, corpus, gold, model = resolve_inputs(config)
    check_hashes(preflight["source_hashes"])
    check_hashes({model / name: sha for name, sha in preflight["model_hashes"].items()})
    if digest(snapshot / "source_snapshot_manifest.json") != preflight["snapshot_manifest_sha256"]:
        raise ValueError("Frozen upstream snapshot differs from V1")
    current = environment()
    current["packages"].update({name: importlib.metadata.version(name)
                                for name in preflight["environment"]["packages"]})
    if current["python"] != preflight["environment"]["python"] or current["packages"] != preflight["environment"]["packages"]:
        raise ValueError("V1 environment versions differ; cannot reuse retrieval")
    if digest(corpus) != preflight["corpus_sha256"] or digest(gold) != preflight["gold_sha256"]:
        raise ValueError("V1 inputs differ")
    return baseline, preflight, current, immutable


def development_examples(config):
    """Small frozen examples from original train documents, no retrieval labels."""
    import pyarrow.parquet as pq
    from parsers.markdown import MarkdownParser

    path = (ARTICLE / config["development"]["qa"]).resolve()
    if digest(path) != config["development"]["qa_sha256"]:
        raise ValueError("Train QA hash mismatch")
    qa = pq.read_table(path).to_pylist()
    development_ids = set()
    for row in qa:
        raw = row.get("ground_truths_contexts_ids", row["correct_answer_document_ids"])
        values = json.loads(raw) if isinstance(raw, str) and raw.lstrip().startswith("[") else raw
        development_ids.update([values] if isinstance(values, str) else values)
    if len(qa) != 45:
        raise ValueError("Unexpected development population")
    labels = [json.loads(line) for line in (ARTICLE / config["semantic_labels"]).read_text().splitlines()]
    selected = [r for r in labels if r["record_id"] in {"pilot_025", "pilot_004", "pilot_008", "pilot_010"}]
    if any(r["doc_id"] not in development_ids for r in selected):
        raise ValueError("Rule development example is not an original-train document")
    examples = [{**r, "label_provenance": "existing_AI_only_" + r["review_status"],
                 "human_adjudicated": False} for r in selected]
    # These two decisions are NEW AI proposals, not human/independent labels.
    doc_id = config["development"]["anchor_document"]
    if doc_id not in development_ids:
        raise ValueError("New example outside development documents")
    corpus = pq.read_table(ARTICLE / config["dataset"]["corpus"], columns=["doc_id", "md_document"]).to_pylist()
    row = next(r for r in corpus if r["doc_id"] == doc_id)
    document = MarkdownParser().parse_text(row["md_document"], document_id=doc_id, source_label="development")
    anchors = [r for r in selected if r["record_id"] == "pilot_025"]
    for name in config["development"]["new_proposal_items"]:
        from article_research.context_packing import plain
        b = next(b for b in document.blocks if plain(b.text) == "- " + name)
        for number, a in enumerate(anchors):
            examples.append({
                "record_id": "ai_v2_" + name.lower().replace(" ", "_"), "doc_id": doc_id,
                "b_block_id": b.block_id, "b_span": [b.char_start, b.char_end], "b_quote": b.text,
                "a_block_id": a["a_block_id"], "a_span": a["a_span"], "a_quote": a["a_quote"],
                "decision": "positive", "dependency_group": "ai_v2_" + name.lower().replace(" ", "_"),
                "edge_id": f"ai_v2_{name}:{number}", "relation_type": "intro_item_scope",
                "annotator_id": "AI_Codex", "review_status": "AI_proposal",
                "label_provenance": "new_development_AI_proposal", "human_adjudicated": False,
                "missing_context": "The field does not name the connection or the creation operation.",
                "rationale": "The operation heading names Watson Query; the intro specifies required connection details. "
                             "Other list fields are not semantically required, but remain in the continuous interval.",
            })
    return {"development_document_ids": sorted(development_ids), "train_questions": len(qa),
            "examples": examples, "human_adjudicated": False, "new_positive_groups": 2,
            "selection": "fixed named original-train examples; no 118-query rankings or outcomes"}


def worker(args):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from article_research.packing_evaluation import (
        document_metrics,
        evidence_metrics,
        select_ranked_context,
    )

    output = args.output_dir.resolve()
    config = read_json(output / "frozen_config.json")
    preflight = read_json(output / "preflight.json")
    if digest(output / "frozen_config.json") != preflight["config_sha256"]:
        raise ValueError("Frozen configuration changed")
    for path, expected in preflight["source_hashes"].items():
        if digest(path) != expected:
            raise ValueError(f"Code changed after preflight: {path}")
    _, _, _, model_path = resolve_inputs(config)
    for name, expected in preflight["model_hashes"].items():
        if digest(model_path / name) != expected:
            raise ValueError(f"Pinned model changed: {name}")
    destination = output / args.worker
    destination.mkdir()
    with RssSampler() as rss:
        import torch
        from article_research.context_packing import (
            ContextAwarePackingChunker,
            ContextPackingConfig,
        )
        from chunking.fixed import TokenizerConfig
        from enrichment.chunks import enrich_chunks
        from retrieval.ranking import (
            BM25Index,
            DenseIndex,
            reciprocal_rank_fusion,
            resolve_text_budget,
        )
        from sentence_transformers import SentenceTransformer

        torch.set_num_threads(config["model"]["cpu_threads"])
        torch.manual_seed(config["seed"])
        np.random.seed(config["seed"])
        documents, sources, gold, _ = load_documents(config, smoke=args.smoke)
        stages = {}
        start = time.perf_counter()
        model = SentenceTransformer(str(model_path), device="cpu", local_files_only=True)
        stages["model_load_seconds"] = time.perf_counter() - start
        budget = resolve_text_budget(model, requested_max_tokens=config["model"]["requested_max_tokens"])
        if budget < 480:
            raise ValueError("Encoder cannot support the frozen target")
        tokenizer_config = TokenizerConfig(name=config["model"]["name"], revision=config["model"]["revision"])
        parameters = config["variants"][args.worker]
        chunker = ContextAwarePackingChunker(ContextPackingConfig(
            tokenizer=tokenizer_config, max_tokens=budget, **parameters), tokenizer=model.tokenizer)
        if budget != preflight["baseline_encoder_text_budget"]:
            raise ValueError("Runtime encoder budget differs from reused v1")
        start = time.perf_counter()
        by_doc, diagnostics, links, candidates = {}, [], [], []
        for doc in documents:
            by_doc[doc.doc_id] = chunker.chunk(doc)
            diagnostics.extend({"doc_id": doc.doc_id, **entry} for entry in getattr(chunker, "diagnostics", []))
            links.extend({"doc_id": doc.doc_id, **entry} for entry in chunker.links)
            candidates.extend({"doc_id": doc.doc_id, **entry} for entry in chunker.candidates)
        stages["segmentation_seconds"] = time.perf_counter() - start
        write_jsonl(destination / "preferred_links.jsonl", links)
        write_jsonl(destination / "intro_candidates.jsonl", candidates)
        start = time.perf_counter()
        validation = [validate_chunks(doc, by_doc[doc.doc_id], model.tokenizer) for doc in documents]
        chunks = [chunk for doc in documents for chunk in enrich_chunks(by_doc[doc.doc_id], doc, sources[doc.doc_id])]
        if any(chunk.token_count > budget for chunk in chunks):
            raise ValueError("Encoder text-budget violation")
        feasible_broken = [r for r in links if r["kind"] == "heading"
                           and r["fits_encoder"] and not r["preserved_in_index"]]
        if feasible_broken:
            raise ValueError("A feasible preferred heading interval was broken")
        known_path = ARTICLE / config["heading_audit"] / "heading_cases.jsonl"
        known = [json.loads(line) for line in known_path.read_text().splitlines()]
        known = [r for r in known if r["first_subtree_body"] is not None and r["doc_id"] in by_doc]
        restored = sum(any(c.char_start <= r["heading_run"][0]["char_start"]
                           and c.char_end >= r["first_subtree_body"]["char_end"] for c in by_doc[r["doc_id"]])
                       for r in known)
        if restored != len(known) or (not args.smoke and len(known) != 132):
            raise ValueError("Known 132 heading/body behavior checks failed")
        write_json(destination / "structural_integrity.json", {
            "known_cases_present": len(known), "known_cases_preserved": restored,
            "feasible_heading_links_broken": len(feasible_broken),
            "checked_before_indexing": True,
        })
        stages["integrity_validation_seconds"] = time.perf_counter() - start
        pq.write_table(pa.Table.from_pylist([chunk.model_dump(mode="json") for chunk in chunks]),
                       destination / "chunks.parquet", compression="zstd")
        (destination / "heading_diagnostics.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in diagnostics), encoding="utf-8")
        start = time.perf_counter()
        bm25 = BM25Index(chunks)
        stages["bm25_build_seconds"] = time.perf_counter() - start
        start = time.perf_counter()
        dense = DenseIndex(chunks, model, batch_size=config["model"]["batch_size"])
        stages["dense_build_including_encoder_validation_seconds"] = time.perf_counter() - start
        print(f"{args.worker}: index built; {len(chunks)} chunks; budget={budget}", flush=True)
        chunk_index = {chunk.chunk_id: index for index, chunk in enumerate(chunks)}
        rank_schema = pa.schema([
            ("query_id", pa.string()), ("method", pa.string()), ("rank", pa.int32()),
            ("chunk_index", pa.int32()), ("score", pa.float64()),
        ])
        metrics = []
        start = time.perf_counter()
        with pq.ParquetWriter(destination / "rankings.parquet", rank_schema, compression="zstd") as writer, \
                (destination / "contexts.jsonl").open("w", encoding="utf-8") as context_file:
            for index, question in enumerate(gold, 1):
                question_text = question["question"]
                sparse, semantic = bm25.search(question_text), dense.search(question_text)
                hybrid = reciprocal_rank_fusion(sparse, semantic, constant=config["retrieval"]["fusion_constant"],
                                               depth=config["retrieval"]["fusion_depth"])
                for method, ranking in [("bm25", sparse), ("dense", semantic), ("hybrid", hybrid)]:
                    spans = [(s["char_start"], s["char_end"]) for s in question["gold_spans"]]
                    metrics.append({
                        "variant": args.worker, "method": method, "query_id": question["question_id"],
                        "gold_doc_id": question["doc_id"],
                        **evidence_metrics(ranking, question["doc_id"], spans),
                        **document_metrics(ranking, [question["doc_id"]]),
                    })
                    selected, used = select_ranked_context(ranking, 1024)
                    context_file.write(json.dumps({
                        "query_id": question["question_id"], "method": method, "used_tokens": used,
                        "chunks": [{"chunk_id": item.chunk.chunk_id, "doc_id": item.chunk.doc_id,
                                    "char_start": item.chunk.char_start, "char_end": item.chunk.char_end,
                                    "token_count": item.chunk.token_count, "rank": item.rank} for item in selected],
                    }) + "\n")
                    n = len(ranking)
                    writer.write_table(pa.Table.from_pydict({
                        "query_id": [question["question_id"]] * n, "method": [method] * n,
                        "rank": [item.rank for item in ranking],
                        "chunk_index": [chunk_index[item.chunk.chunk_id] for item in ranking],
                        "score": [item.score for item in ranking],
                    }, schema=rank_schema))
                if index % 20 == 0 or index == len(gold):
                    print(f"{args.worker}: {index}/{len(gold)} questions", flush=True)
        stages["query_ranking_scoring_and_artifact_io_seconds"] = time.perf_counter() - start
        pq.write_table(pa.Table.from_pylist(metrics), destination / "per_question_metrics.parquet", compression="zstd")
        lengths = [chunk.token_count for chunk in chunks]
        overlaps = []
        for doc in documents:
            right = 0
            for chunk in by_doc[doc.doc_id]:
                end = min(right, chunk.char_end)
                overlaps.append(len(model.tokenizer(doc.canonical_text[chunk.char_start:end],
                                add_special_tokens=False, truncation=False)["input_ids"]) if end > chunk.char_start else 0)
                right = max(right, chunk.char_end)
        stats = {
            "documents": len(documents), "questions": len(gold), "chunk_count": len(chunks),
            "indexed_tokens": sum(lengths), "one_chunk_documents": sum(len(c) == 1 for c in by_doc.values()),
            "chunk_tokens": {"min": min(lengths), "p10": float(np.quantile(lengths, .1)),
                             "median": float(np.median(lengths)), "p90": float(np.quantile(lengths, .9)), "max": max(lengths)},
            "actual_overlap_tokens_sum": sum(overlaps), "actual_overlap_tokens_max": max(overlaps),
            "fallback_chunks": sum(chunk.fallback_split for chunk in chunks),
            "encoder_text_budget": budget, "dense_matrix_bytes": dense.vectors.nbytes,
            "dense_matrix_shape": list(dense.vectors.shape), "dense_matrix_dtype": str(dense.vectors.dtype),
            "uncovered_characters": sum(row["uncovered_characters"] for row in validation),
            "uncovered_nonwhitespace_characters": sum(row["uncovered_nonwhitespace_characters"] for row in validation),
            "heading_diagnostic_reasons": dict(Counter(row["reason"] for row in diagnostics)),
            "stages_seconds": stages, "chunker_provenance": chunker.provenance,
            "pid": os.getpid(), "rss_sampling_seconds": .1,
            "preferred_links_by_kind": dict(Counter(r["kind"] for r in links)),
            "intro_candidates_by_decision": dict(Counter(r["decision"] for r in candidates)),
        }
    stats["peak_process_rss_bytes"] = rss.peak
    stats["rss_scope"] = "isolated_variant_process_including_model_and_index_sampled_100ms"
    write_json(destination / "index_statistics.json", stats)
    write_json(destination / "manifest.json", {
        "status": "completed", "finished_at_utc": now(), "variant": args.worker,
        "config_sha256": preflight["config_sha256"], "question_count": len(gold),
        "artifact_sha256": {p.name: digest(p) for p in sorted(destination.iterdir()) if p.is_file()},
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ARTICLE / "configs/context_packing_v2.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--worker", choices=["headings", "headings_intro"])
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    from run_stage_a_baseline import validate_run_dir
    output = validate_run_dir(args.output_dir, ARTICLE)
    config = read_json(args.config)
    baseline, old_preflight, versions, immutable = verify_baseline(config)
    snapshot, corpus, gold_path, _model = resolve_inputs(config)
    examples = development_examples(config)
    # Known cases are behavior checks, never a semantic rule-development set.
    audit = (ARTICLE / config["heading_audit"]).resolve()
    immutable.update(verified_artifacts(audit))
    labels = (ARTICLE / config["semantic_labels"]).resolve()
    immutable[str(labels)] = digest(labels)
    output.mkdir(parents=True)
    write_json(output / "frozen_config.json", config)
    write_json(output / "development_examples.json", examples)
    source_files = [Path(__file__).resolve(), ARTICLE / "src/article_research/context_packing.py",
                    ARTICLE / "tests/test_context_packing.py", ARTICLE / "docs/context_packing_v2_protocol.md"]
    source_hashes = {**old_preflight["source_hashes"], **{str(p): digest(p) for p in source_files}}
    (output / "requirements.lock.txt").write_text(subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze"], text=True), encoding="utf-8")
    write_json(output / "preflight.json", {
        "status": "passed_frozen_before_new_retrieval", "frozen_at_utc": now(), "smoke": args.smoke,
        "config_sha256": digest(output / "frozen_config.json"), "source_hashes": source_hashes,
        "model_hashes": old_preflight["model_hashes"], "environment": versions,
        "corpus_sha256": digest(corpus), "gold_sha256": digest(gold_path),
        "snapshot_id": snapshot.name, "snapshot_manifest_sha256": digest(snapshot / "source_snapshot_manifest.json"),
        "development_examples_sha256": digest(output / "development_examples.json"),
        "development_qa_sha256": config["development"]["qa_sha256"],
        "semantic_labels_sha256": digest(labels), "immutable_source_artifacts": immutable,
        "baseline_run": str(baseline), "baseline_variant": "hierarchy",
        "baseline_reuse": "passed_input_model_source_environment_and_protocol_checks",
        "baseline_encoder_text_budget": read_json(baseline / "hierarchy/index_statistics.json")["encoder_text_budget"],
        "exact_code_state": "sha256_all_executed_sources_and_validated_frozen_snapshot; git unavailable",
        "git_metadata": old_preflight.get("git_states"), "command": sys.argv,
        "platform": platform.platform(), "seed": config["seed"],
    })
    write_json(output / "v1_reuse.json", {"source": str(baseline), "variant": "hierarchy",
               "status": "validated_reuse" if not args.smoke else "validated_reference_full_v1_not_paired_with_smoke",
               "manifest_sha256": digest(baseline / "manifest.json"), "costs_are_historical_v1_measurements": True})
    for variant in config["variants"]:
        command = [sys.executable, str(Path(__file__).resolve()), "--output-dir", str(output), "--worker", variant]
        if args.smoke:
            command.append("--smoke")
        result = subprocess.run(command, check=False, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        if result.returncode:
            write_json(output / "failure.json", {"variant": variant, "returncode": result.returncode, "time": now()})
            raise RuntimeError(f"V2 failed; no substituted results: {variant}")
    check_hashes(immutable)
    check_hashes(source_hashes)
    write_json(output / "retrieval_manifest.json", {
        "status": "completed_smoke" if args.smoke else "completed_new_exploratory_retrieval",
        "finished_at_utc": now(), "old_artifacts_unchanged": True,
        "artifact_sha256": {str(p.relative_to(output)): digest(p) for p in sorted(output.rglob("*")) if p.is_file()},
    })
    print(json.dumps({"output": str(output), "status": "retrieval_completed", "smoke": args.smoke}), flush=True)


if __name__ == "__main__":
    main()

