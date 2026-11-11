"""Independent coordinate coverage and budget replay of the two new V2 variants."""
from pathlib import Path
import csv
import hashlib
import json
import statistics
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ARTICLE = HERE.parents[2]
RUN = ARTICLE / 'results/runs/context_full_20261027_v2'
GOLD = ARTICLE.parent / 'techdoc-chunking-pipeline/article/results/stage_a/step5a_ai_approved_freeze_v2/retrieval_gold_ai.jsonl'
gold = {q['question_id']: q for q in map(json.loads, GOLD.read_text().splitlines())}


def coverage(q, chunks):
    hit = total = 0
    for g in q['gold_spans']:
        a, b = g['char_start'], g['char_end']
        total += b - a
        right = a
        intervals = sorted((max(a, c['char_start']), min(b, c['char_end']))
                           for c in chunks if c['doc_id'] == q['doc_id'])
        for x, y in intervals:
            if y > max(x, right):
                hit += y - max(x, right)
                right = y
    return hit / total


expected = {(r['variant'], r['method'], r['query_id']): r
            for r in pq.read_table(RUN / 'per_question_metrics.parquet').to_pylist()}
aggregates = {(r['variant'], r['method']): r for r in csv.DictReader((RUN / 'aggregate_metrics.csv').open())}
rows, inputs = [], {}
for variant in ['headings', 'headings_intro']:
    folder = RUN / variant
    for name in ['chunks.parquet', 'rankings.parquet']:
        inputs[str(folder / name)] = hashlib.sha256((folder / name).read_bytes()).hexdigest()
    chunks = pq.read_table(folder / 'chunks.parquet').to_pylist()
    rankings = pq.read_table(folder / 'rankings.parquet').to_pandas()
    contexts = {(c['query_id'], c['method']): c
                for c in map(json.loads, (folder / 'contexts.jsonl').read_text().splitlines())}
    for (qid, method), r in rankings.groupby(['query_id', 'method']):
        ranked = [chunks[i] for i in r.sort_values('rank').chunk_index]
        selected, used = [], 0
        for c in ranked:
            if used + c['token_count'] <= 1024:
                selected.append(c)
                used += c['token_count']
        stored_context = contexts[qid, method]
        assert used == stored_context['used_tokens']
        assert [c['chunk_id'] for c in selected] == [c['chunk_id'] for c in stored_context['chunks']]
        row = {'variant': variant, 'method': method, 'query_id': qid,
               'EvidenceCoverage@B=1024': coverage(gold[qid], selected),
               'EvidenceCoverage@5': coverage(gold[qid], ranked[:5])}
        for metric in ['EvidenceCoverage@B=1024', 'EvidenceCoverage@5']:
            assert abs(row[metric] - expected[variant, method, qid][metric]) < 1e-12
            assert row[metric] == expected['v1', method, qid][metric]
        rows.append(row)
    for method in ['bm25', 'dense', 'hybrid']:
        for metric in ['EvidenceCoverage@B=1024', 'EvidenceCoverage@5']:
            value = statistics.mean(r[metric] for r in rows if r['variant'] == variant and r['method'] == method)
            assert abs(value - float(aggregates[variant, method][metric])) < 1e-12
result = {'status': 'passed', 'independent_new_v2_rankings_replayed': len(rows),
          'coverage_values_recomputed': 2 * len(rows),
          'all_new_v2_b1024_context_selections_match': True,
          'all_new_v2_coverage_values_equal_v1': True,
          'input_sha256': inputs,
          'scope': 'Saved chunks/rankings coordinate replay; no tokenization, encoder or retrieval rerun'}
(HERE / 'v2_coverage_replay.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: v for k, v in result.items() if k != 'input_sha256'}))
