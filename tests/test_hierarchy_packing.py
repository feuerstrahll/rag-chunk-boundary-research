from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ARTICLE = Path(__file__).resolve().parents[1]
SNAPSHOT = ARTICLE / "data/intermediate/upstream_snapshot/7f2f09d07dbd925fcdcdc6e6c5b564d355fc287aa1e5eccb7da9633648a9cd89"
sys.dont_write_bytecode = True
sys.path.insert(0, str(SNAPSHOT / "src"))
sys.path.insert(0, str(ARTICLE / "src"))

from article_research.hierarchy_packing import (
    HierarchyAwarePackingChunker,
    HierarchyPackingConfig,
)
from article_research.packing_evaluation import (
    document_clusters,
    document_metrics,
    evidence_metrics,
    paired_cluster_bootstrap,
    select_ranked_context,
)
from chunking.fixed import TokenizerConfig
from models import Block, Document


class Words:
    is_fast = True
    _commit_hash = "local-test-tokenizer"

    def __call__(self, text, **kwargs):
        matches = list(re.finditer(r"\S+", text))
        return {"input_ids": list(range(len(matches))), "offset_mapping": [m.span() for m in matches]}


def doc(entries):
    return Document(doc_id="d", source="fixture", blocks=[
        Block(block_id=f"b{i}", type="heading" if level else "paragraph", text=text, level=level)
        for i, (level, text) in enumerate(entries)
    ])


def pack(document, *, hierarchy=True, target=10, hard=14, overlap=2):
    chunker = HierarchyAwarePackingChunker(HierarchyPackingConfig(
        tokenizer=TokenizerConfig(name="fixture", revision="local-test-tokenizer"),
        target_tokens=target, max_tokens=hard, overlap_tokens=overlap,
        respect_parent_boundary=hierarchy,
    ), tokenizer=Words())
    chunks = chunker.chunk(document)
    for chunk in chunks:
        assert chunk.text == document.canonical_text[chunk.char_start:chunk.char_end]
        assert chunk.token_count <= hard
    for match in re.finditer(r"\S", document.canonical_text):
        assert any(c.char_start <= match.start() < c.char_end for c in chunks)
    return chunker, chunks


def spans(chunks):
    return [(c.char_start, c.char_end) for c in chunks]


def test_sibling_h2_allowed_under_real_h1():
    d = doc([(1, "root"), (2, "a"), (None, "body a"), (2, "b"), (None, "body b")])
    _, chunks = pack(d)
    assert any("a\n\nbody a\n\nb\n\nbody b" in c.text for c in chunks)


def test_repeated_root_names_are_distinct_instances():
    d = doc([(1, "same"), (None, "one"), (1, "same"), (None, "two")])
    _, chunks = pack(d)
    assert len(chunks) == 2
    _, unrestricted = pack(d, hierarchy=False)
    assert len(unrestricted) == 1


def test_parent_changes_prevent_h3_merge_and_overlap():
    d = doc([(1, "r"), (2, "same"), (3, "a"), (None, "one"), (2, "same"), (3, "b"), (None, "two")])
    _, chunks = pack(d)
    assert not any("one" in c.text and "two" in c.text for c in chunks)
    assert all(a.char_end <= b.char_start for a, b in zip(chunks, chunks[1:]))


def test_skipped_levels_use_nearest_lower_heading():
    d = doc([(1, "r"), (3, "a"), (None, "one"), (3, "b"), (None, "two")])
    _, chunks = pack(d)
    assert any("one" in c.text and "two" in c.text for c in chunks)


def test_no_headings_identical_ablation():
    d = doc([(None, "a b c d e f"), (None, "g h i j k l"), (None, "m n")])
    assert spans(pack(d)[1]) == spans(pack(d, hierarchy=False)[1])


@pytest.mark.parametrize("hierarchy", [True, False])
def test_heading_moves_with_body_at_target_boundary(hierarchy):
    d = doc([(None, "a b c d e f g h i"), (2, "title"), (None, "body words")])
    _, chunks = pack(d, hierarchy=hierarchy)
    assert chunks[0].text.endswith("i")
    assert any("title\n\nbody words" in c.text for c in chunks)
    assert not any(c.text.endswith("title") for c in chunks)


@pytest.mark.parametrize("hierarchy", [True, False])
def test_heading_body_can_exceed_target_but_not_hard_limit(hierarchy):
    d = doc([(2, "heading"), (None, " ".join(["body"] * 12))])
    _, chunks = pack(d, hierarchy=hierarchy)
    assert len(chunks) == 1
    assert chunks[0].token_count == 13
    assert not chunks[0].fallback_split


@pytest.mark.parametrize("hierarchy", [True, False])
def test_oversized_heading_body_uses_existing_fallback(hierarchy):
    d = doc([(2, "heading"), (None, " ".join(["body"] * 30))])
    chunker, chunks = pack(d, hierarchy=hierarchy)
    assert all(c.fallback_split for c in chunks)
    assert chunks[0].text.startswith("heading\n\nbody")
    assert chunks[1].char_start < chunks[0].char_end
    assert chunker.diagnostics[0]["reason"] == "heading_body_oversized_fallback"


def test_long_heading_and_empty_end_have_explicit_diagnostics():
    d = doc([(2, " ".join(["heading"] * 20)), (None, "body"), (2, "empty")])
    chunker, _ = pack(d)
    reasons = {x["reason"] for x in chunker.diagnostics}
    assert "heading_exceeds_limit" in reasons
    assert "no_following_content" in reasons


def test_real_size_edges_and_comments_consume_tokens():
    for size in [480, 481, 510, 511]:
        d = doc([(None, " ".join(["<!--comment-->"] * size))])
        _, chunks = pack(d, target=480, hard=510, overlap=64)
        assert len(chunks) == (1 if size <= 510 else 2)
        assert chunks[0].token_count == min(size, 510)


def test_overlap_counts_towards_target_and_can_shrink():
    d = doc([(None, "a b c d e f g h"), (None, "i j"), (None, "k l m n o p q r s")])
    _, chunks = pack(d)
    assert len(chunks) == 2
    assert chunks[1].token_count == 9
    assert chunks[1].char_start >= chunks[0].char_end


def ranked(doc_id, a, b, rank, tokens=10, score=None):
    return SimpleNamespace(chunk=SimpleNamespace(doc_id=doc_id, char_start=a, char_end=b, token_count=tokens),
                           rank=rank, score=1 / rank if score is None else score)


def test_document_pooling_before_top_k_and_mrr_cutoff():
    ranking = [ranked("distractor", 0, 1, i) for i in range(1, 21)] + [ranked("gold", 0, 1, 21, score=.04)]
    assert document_metrics(ranking, ["gold"]) == {"DocumentRecall@5": 1., "DocumentMRR@10": .5}
    ranking = [ranked(str(i), 0, 1, i + 1) for i in range(11)]
    assert document_metrics(ranking, ["10"])["DocumentMRR@10"] == 0


def test_evidence_union_hit_differs_from_single_chunk_mrr():
    ranking = [ranked("gold", 0, 60, 1), ranked("gold", 40, 100, 2)]
    result = evidence_metrics(ranking, "gold", [(0, 100)])
    assert result["EvidenceCoverage@5"] == 1
    assert result["EvidenceRecall@5"] == 1
    assert result["EvidenceMRR@5"] == 0
    assert result["actual_context_tokens_B1024"] == 20


def test_budget_skips_nonfitting_without_truncation():
    ranking = [ranked("a", 0, 1, 1, tokens=700), ranked("a", 0, 1, 2, tokens=400), ranked("b", 0, 1, 3, tokens=300)]
    selected, used = select_ranked_context(ranking, 1024)
    assert [r.rank for r in selected] == [1, 3]
    assert used == 1000


def test_multi_document_connected_components_are_transitive():
    groups = document_clusters({"q1": ["a", "b"], "q2": ["b", "c"], "q3": ["d"], "q4": ["c"]})
    assert groups["q1"] == groups["q2"] == groups["q4"]
    assert groups["q3"] != groups["q1"]


def test_cluster_bootstrap_samples_related_questions_jointly():
    reference = {"a": 0., "b": 0., "c": 0.}
    candidate = {"a": 1., "b": -1., "c": 0.}
    result = paired_cluster_bootstrap(reference, candidate, {"a": "x", "b": "x", "c": "y"})
    assert result["cluster_count"] == 2
    assert result["ci_lower"] == result["ci_upper"] == result["difference"] == 0.
    with pytest.raises(ValueError):
        paired_cluster_bootstrap(reference, {"a": 0}, {"a": "x"})


def test_metrics_equal_existing_step5b_definitions(monkeypatch):
    upstream = ARTICLE.parent / "techdoc-chunking-pipeline/article/src/article_research"
    def load(name, filename):
        spec = importlib.util.spec_from_file_location(name, upstream / filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    reference_stage4 = load("reference_stage4", "stage4.py")
    monkeypatch.setitem(sys.modules, "article_research.stage4", reference_stage4)
    reference_step5b = load("reference_step5b", "step5b.py")
    ranking = [ranked("gold", 0, 45, 1, tokens=600), ranked("other", 0, 100, 2, tokens=500),
               ranked("gold", 45, 100, 3, tokens=300)]
    expected = reference_step5b.question_metrics(ranking, "gold", [(0, 40), (60, 100)], top_k=5, threshold=.8, budget=1024)
    assert evidence_metrics(ranking, "gold", [(0, 40), (60, 100)]) == expected


def test_four_variant_summary_preserves_retriever_and_renders_reports(tmp_path):
    """Exercise final delivery as well as the per-question metric functions."""
    import json

    import pyarrow as pa
    import pyarrow.parquet as pq
    sys.path.insert(0, str(ARTICLE / "scripts"))
    import run_hierarchy_packing as runner
    from article_research.packing_evaluation import METRICS

    config = json.loads((ARTICLE / "configs/hierarchy_packing.json").read_text())
    gold = [{"question_id": "q", "doc_id": "d"}]
    for variant in config["variants"]:
        folder = tmp_path / variant
        folder.mkdir()
        rows = [{"variant": variant, "method": method, "query_id": "q", "gold_doc_id": "d",
                 **{name: .5 for name in METRICS}, "actual_context_tokens_B1024": 500}
                for method in ["bm25", "dense", "hybrid"]]
        pq.write_table(pa.Table.from_pylist(rows), folder / "per_question_metrics.parquet")
        pq.write_table(pa.Table.from_pylist([{"doc_id": "d", "char_start": 0, "char_end": 1}]), folder / "chunks.parquet")
        stats = {"chunk_count": 1, "indexed_tokens": 10, "chunk_tokens": {"p10": 10, "median": 10, "p90": 10},
                 "actual_overlap_tokens_sum": 0, "peak_process_rss_bytes": 1000, "dense_matrix_bytes": 100,
                 "stages_seconds": {name: 0.1 for name in ["segmentation_seconds", "bm25_build_seconds",
                      "dense_build_including_encoder_validation_seconds"]}}
        (folder / "index_statistics.json").write_text(json.dumps(stats))
    summary = runner.summarize(tmp_path, config, [doc([(None, "body")])], gold)
    assert summary["per_question_rows"] == 12
    assert {row["method"] for row in summary["paired_intervals"]} == {"bm25", "dense", "hybrid"}
    assert (tmp_path / "report.md").is_file()
    assert (tmp_path / "article_results.md").is_file()
