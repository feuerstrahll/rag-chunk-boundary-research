import json, collections
from pathlib import Path
import pandas as pd
R=Path('article/results/runs'); O=Path('article/results/reviews/article_recheck_20261027_v2/new_results')
def jl(p):return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
vroot=R/'context_full_20261027_v2'; hroot=R/'hierarchy_full_20261026_v1'
paths={'v1':hroot/'hierarchy','headings':vroot/'headings','headings_intro':vroot/'headings_intro'}
chunks={v:pd.read_parquet(p/'chunks.parquet') for v,p in paths.items()}
out={}
out['chunks']={v:{'rows':len(d),'docs':d.doc_id.nunique(),'tokens':int(d.token_count.sum()),'quantiles':d.token_count.quantile([0,.1,.5,.9,1]).to_dict(),'fallback':int(d.fallback_split.sum()),'dense_MiB':len(d)*384*4/1024**2} for v,d in chunks.items()}
sigs={v:{doc:tuple(x[['char_start','char_end','chunk_id']].itertuples(index=False,name=None)) for doc,x in d.groupby('doc_id',sort=False)} for v,d in chunks.items()}
out['changed_docs']={v:sum(sig!=sigs['v1'][doc] for doc,sig in s.items()) for v,s in sigs.items()}
out['v2_identical']=sigs['headings']==sigs['headings_intro']
metrics=pd.read_parquet(vroot/'per_question_metrics.parquet')
out['metrics_means']=metrics.groupby(['variant','method']).mean(numeric_only=True).reset_index().to_dict('records')
out['paired_changes']={}
for method in ['bm25','dense','hybrid']:
 a=metrics[(metrics.variant=='v1')&(metrics.method==method)].set_index('query_id');b=metrics[(metrics.variant=='headings')&(metrics.method==method)].set_index('query_id').loc[a.index]
 out['paired_changes'][method]={c:int((a[c]!=b[c]).sum()) for c in a.select_dtypes('number').columns}
 out['paired_changes'][method]['changed_gold_documents']=int(a.gold_doc_id.isin([k for k,s in sigs['v1'].items() if sigs['headings'][k]!=s]).sum())
groups=jl(vroot/'evaluation_groups.jsonl')
links=jl(vroot/'headings'/'preferred_links.jsonl')
def contains(ch, spans):return all(ch['char_start']<=a and ch['char_end']>=b for a,b in spans)
bydoc={v:{doc:x.to_dict('records') for doc,x in d.groupby('doc_id')} for v,d in chunks.items()}
out['individual_heading_links']={v:{'total':len(links),'preserved':sum(any(contains(ch, g['a_spans']+[g['b_span']]) for ch in bydoc[v][g['doc_id']]) for g in links)} for v in chunks}
out['groups_preserved']={v:{co:{'total':sum(g['cohort']==co for g in groups),'preserved':sum(any(contains(ch,g['a_spans']+[g['b_span']]) for ch in bydoc[v][g['doc_id']]) for g in groups if g['cohort']==co)} for co in sorted(set(g['cohort'] for g in groups))} for v in chunks}
# Recompute conditional completeness by union of retrieved canonical spans, no labels imported.
def union_contains(chs,span):
 a,b=span
 for x,y in sorted((x['char_start'],x['char_end']) for x in chs):
  if x<=a: a=max(a,y)
  if a>=b:return True
 return False
qa=metrics[(metrics.variant=='v1')&(metrics.method=='hybrid')].set_index('query_id').gold_doc_id.to_dict()
out['completeness']={}
for v,p in paths.items():
 contexts=[c for c in jl(p/'contexts.jsonl') if c['method']=='hybrid'];co=collections.Counter();miss=[]
 for c in contexts:
  gs=[g for g in groups if g['cohort']=='structural_chain' and g['doc_id']==qa[c['query_id']]]
  co['eligible']+=len(gs)
  for g in gs:
   selected=[x for x in c['chunks'] if x['doc_id']==g['doc_id']]
   bf=union_contains(selected,g['b_span']);af=all(union_contains(selected,a) for a in g['a_spans'])
   co['b_full']+=bf;co['complete']+=bf and af
   if bf and not af:miss.append({'query':c['query_id'],'group':g,'chunks':c['chunks'],'used_tokens':c['used_tokens']})
 out['completeness'][v]={'counts':dict(co),'incomplete':miss}
# Raw model judgments joined to prepared request mapping.
prep=R/'context_sufficiency_20261027_v2'; run=R/'context_sufficiency_openrouter_20261027_v2'
mp={r['request_id']:r for r in jl(prep/'mapping.jsonl')};jud=jl(run/'judgments.jsonl');resp=jl(run/'responses.jsonl');req={r['request_id']:r for r in jl(prep/'requests.jsonl')}
assert len(jud)==len({r['request_id'] for r in jud})
assert {r['request_id'] for r in jud}=={r['request_id'] for r in resp}
joined=[mp[j['request_id']]|j for j in jud];jj=pd.DataFrame(joined)
out['judgments']={'prepared':len(mp),'evaluated':len(jud),'distinct_questions':len(set(jj.query_id)),'providers':dict(collections.Counter(r['response'].get('provider') for r in resp)), 'models':dict(collections.Counter(r['response'].get('model') for r in resp)),'rows':[]}
for (c,v),d in jj.groupby(['cohort','variant']):
 out['judgments']['rows'].append({'cohort':c,'variant':v,'n':len(d),'counts':dict(collections.Counter(d.sufficiency)),'SR':float((d.sufficiency=='sufficient').mean()),'coverage_mean':float(d['EvidenceCoverage@1024'].mean()) if c=='AI-only' else None})
out['pairs']=[]
for (c,q),d in jj.groupby(['cohort','query_id']):
 if len(d)==2:out['pairs'].append({'cohort':c,'query':q,'labels':dict(zip(d.variant,d.sufficiency)),'gold_docs':d.iloc[0].gold_document_ids})
review=jl('article/results/analyses/context_sufficiency_review_20261027_v1/manual_content_review.jsonl')
out['review_counts']=dict(collections.Counter(r['assessment'] for r in review))
out['disagreements_with_coverage']=[{'query':r['query_id'],'variant':r['variant'],'coverage':r['EvidenceCoverage@1024'],'label':r['sufficiency']} for r in joined if r['cohort']=='AI-only' and ((r['EvidenceCoverage@1024']==1 and r['sufficiency']!='sufficient')or(r['EvidenceCoverage@1024']==0 and r['sufficiency']=='sufficient'))]
# Model JSON labels exactly match artifact judgments.
out['raw_response_match']=all(json.loads(r['response']['choices'][0]['message']['content'])=={k:j[k] for k in ['sufficiency','reason','missing_information']} for r in resp for j in jud if j['request_id']==r['request_id'])
(O/'recomputed.json').write_text(json.dumps(out,indent=2,ensure_ascii=False))
print(json.dumps({k:v for k,v in out.items() if k not in ['completeness','metrics_means']},ensure_ascii=False,indent=2));print('completeness counts',[(v,x['counts']) for v,x in out['completeness'].items()])
