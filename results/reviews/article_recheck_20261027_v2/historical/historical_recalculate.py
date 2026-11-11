import json,collections,random
from statistics import fmean
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
out=Path(__file__).parent
old={}
for v,p in [('fixed','techdoc-chunking-pipeline/results/watsonx_budget_matched/md_document/fixed_matched/rankings.jsonl'),('structure','techdoc-chunking-pipeline/results/watsonx_retrieval/md_document/structure/rankings.jsonl')]:
 rows=[r for r in map(json.loads,open(p)) if r['method']=='hybrid']; print(v,'metric',rows[0]['evidence_metrics']);old[v]={r['query_id']:r['evidence_metrics']['EvidenceCoverage@5'] for r in rows}
def boot(a,b,keys):
 d=[b[k]-a[k] for k in keys];rng=random.Random(2026);sampled=sorted(fmean(d[rng.randrange(len(d))] for _ in d) for _ in range(10000));return {'fixed':fmean(a[k] for k in keys),'structure':fmean(b[k] for k in keys),'delta':fmean(d),'ci':[sampled[250],sampled[9750]]}
r={'original_75_from_saved_per_question_scores':boot(old['fixed'],old['structure'],sorted(old['fixed']))}
p='techdoc-chunking-pipeline/article/results/stage_a/step5b_20261020_v1/per_question_metrics.parquet'
rows=pq.read_table(p).to_pylist();r['step5b_from_per_question']={}
for m in ['EvidenceCoverage@5','EvidenceCoverage@B=1024']:
 pairs={v:{x['question_id']:x[m] for x in rows if x['variant']==v} for v in ['fixed','structure']};r['step5b_from_per_question'][m]=boot(pairs['fixed'],pairs['structure'],sorted(pairs['fixed']))
audit=pq.read_table('techdoc-chunking-pipeline/article/results/stage_a/gold_audit_20261018_v1/gold_alignment_audit.parquet').to_pylist();r['train_alignment_audit']={'rows':len(audit),'statuses':dict(collections.Counter(x['status'] for x in audit)),'methods':dict(collections.Counter(x['alignment_method'] for x in audit))}
(out/'historical_recalculation.json').write_text(json.dumps(r,ensure_ascii=False,indent=2));print(json.dumps(r,ensure_ascii=False,indent=2))
