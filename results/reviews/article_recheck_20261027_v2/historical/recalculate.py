import json,csv,collections,hashlib
from pathlib import Path
import pyarrow.parquet as pq
import numpy as np
out=Path(__file__).parent
root=Path('article/results/runs/hierarchy_full_20261026_v1')
gold={r['question_id']:r for r in map(json.loads,open('techdoc-chunking-pipeline/article/results/stage_a/step5a_ai_approved_freeze_v2/retrieval_gold_ai.jsonl'))}
def cov(g, chunks):
 total=hit=0
 for s in g['gold_spans']:
  a,b=s['char_start'],s['char_end'];total+=b-a
  intervals=sorted((max(a,c['char_start']),min(b,c['char_end'])) for c in chunks if c['doc_id']==g['doc_id'])
  edge=a
  for x,y in intervals:
   if y>max(x,edge):hit+=y-max(x,edge);edge=y
 return hit/total
rows=[];costs={};differences=[];alls={}
for variant in ['fixed','structure','hierarchy','unrestricted']:
 chunks=pq.read_table(root/variant/'chunks.parquet').to_pylist();alls[variant]=chunks
 ranks=pq.read_table(root/variant/'rankings.parquet').to_pandas()
 stored={(r['query_id'],r['method']):r for r in pq.read_table(root/variant/'per_question_metrics.parquet').to_pylist()}
 for (qid,method),rs in ranks.groupby(['query_id','method'],sort=False):
  rs=rs.sort_values('rank'); ranked=[chunks[i] for i in rs.chunk_index];g=gold[qid]
  selected=[];used=0
  for c in ranked:
   if used+c['token_count']<=1024: selected.append(c);used+=c['token_count']
  ec=cov(g,ranked[:5]); mrr=next((1/i for i,c in enumerate(ranked[:5],1) if cov(g,[c])>=.8),0)
  scores={}
  for c,s in zip(ranked,rs.score):scores[c['doc_id']]=max(scores.get(c['doc_id'],-float('inf')),s)
  docs=sorted(scores,key=lambda d:(-scores[d],d)); dr=next((i for i,d in enumerate(docs,1) if d==g['doc_id']),100000)
  row=dict(variant=variant,method=method,query_id=qid,**{'EvidenceCoverage@5':ec,'EvidenceRecall@5':float(ec>=.8),'EvidenceMRR@5':mrr,'EvidenceCoverage@B=1024':cov(g,selected),'actual_context_tokens_B1024':used,'DocumentRecall@5':float(dr<=5),'DocumentMRR@10':1/dr if dr<=10 else 0})
  for k,v in row.items():
   if isinstance(v,(int,float)) and abs(v-stored[qid,method][k])>1e-12:differences.append((variant,qid,method,k,v,stored[qid,method][k]))
  rows.append(row)
 toks=[c['token_count'] for c in chunks];counts=collections.Counter(c['doc_id'] for c in chunks)
 costs[variant]={'chunks':len(chunks),'tokens':sum(toks),'one_chunk_documents':sum(n==1 for n in counts.values()),'quantiles':np.quantile(toks,[.1,.5,.9]).tolist(),'dense_mib':len(chunks)*384*4/1048576}
 print(variant,'done',len(ranks),flush=True)
metrics=[k for k in rows[0] if k not in ('variant','method','query_id')]
agg=[]
for v in alls:
 for m in ['bm25','dense','hybrid']:
  selected=[r for r in rows if r['variant']==v and r['method']==m]
  agg.append(dict(variant=v,method=m,**{k:float(np.mean([r[k] for r in selected])) for k in metrics}))
clusters=sorted((g['doc_id'],qid) for qid,g in gold.items())
intervals=[]
for m in ['hybrid','bm25','dense']:
 for ref in ['structure','fixed','unrestricted']:
  by={(r['variant'],r['query_id']):r for r in rows if r['method']==m}
  d=np.array([by['hierarchy',qid]['EvidenceCoverage@B=1024']-by[ref,qid]['EvidenceCoverage@B=1024'] for doc,qid in clusters])
  rng=np.random.default_rng(2026);draw=rng.integers(0,len(d),size=(10000,len(d)))
  intervals.append(dict(method=m,contrast='hierarchy_minus_'+ref,difference=float(np.mean(d)),ci=np.quantile(d[draw].mean(axis=1),[.025,.975]).tolist()))
lengths=list(csv.DictReader(open('article/results/analyses/hierarchy_posthoc_20261026_v2/document_lengths.csv')))
length_lookup={r['doc_id']:r for r in lengths};bands={}
for r in rows:
 if r['method']!='hybrid':continue
 b=length_lookup[gold[r['query_id']]['doc_id']]['length_band'];bands.setdefault(b,{}).setdefault(r['variant'],[]).append(r['EvidenceCoverage@B=1024'])
span_eq=[]
for v in ['hierarchy','unrestricted']:
 groups=collections.defaultdict(list)
 for c in alls[v]:groups[c['doc_id']].append((c['char_start'],c['char_end']))
 span_eq.append(groups)
result={'question_count':len(gold),'span_count':sum(len(g['gold_spans']) for g in gold.values()),'recomputed_rows':len(rows),'per_question_mismatches':differences,'costs_from_chunks':costs,'aggregate_from_rankings':agg,'intervals_independent':intervals,'length_bands_from_saved_lengths':{b:{v:{'n':len(x),'mean':float(np.mean(x))} for v,x in vs.items()} for b,vs in bands.items()},'span_identical_docs':sum(sorted(span_eq[0][d])==sorted(span_eq[1][d]) for d in span_eq[0]),'length_quantiles_from_saved_lengths':np.quantile([int(r['canonical_tokens']) for r in lengths],[.5,.75,.9,1]).tolist()}
(out/'independent_recalculation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps(result,ensure_ascii=False,indent=2))
