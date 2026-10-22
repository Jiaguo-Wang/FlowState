"""不调用选择器，独立以十进制公式复核全部子集和策略结果。"""
import hashlib,itertools,json
from decimal import Decimal
from pathlib import Path
P=Path(__file__).resolve().parent
r=json.loads((P/'results.json').read_text());ex=json.loads((P/'exact_verification.json').read_text())
raw=json.loads((P.parent/'multishare_runtime_gate_b5c3/strict_epoch_snapshot.json').read_text())
cs={c['checkpoint_id']:c for c in raw['candidates']};ps=raw['pendings']
edges={(e['checkpoint_id'],e['continuation_id']):e['status'] for e in raw['edges']}
m=r['cost_model'];a,b,c=[Decimal(str(m['coefficient_'+k])) for k in 'abc']
def phi(g,t):
 g=Decimal(g)/1024;t=Decimal(t)/1024
 return a*g+b*g*t+c*g*g

def score(selected):
 result=[]
 for p in ps:
  e=max([cs[x]['token_position_L'] for x in selected if edges[x,p['continuation_id']]=='EPOCH_ADMISSIBLE']+[0]);t=p['planning_target_T']
  result.append((e,t-e,phi(t-e,t)))
 return result
allcost={}
for x in ex['all_subsets']:
 z=score(x['selected']);value=sum(v[2] for v in z);allcost[tuple(x['selected'])]=value
 assert abs(float(value)-x['J_ms'])<1e-9
 for actual,expected in zip(x['per_pending'],z):assert (actual['E'],actual['G'])==expected[:2]
for x in r['rows']:
 z=score(x['selected']);assert abs(float(sum(v[2] for v in z))-x['J_ms'])<1e-9
 assert len(set(x['selected']))==len(x['selected'])<=x['K']
 best=min(v for s,v in allcost.items() if len(s)<=x['K'])
 assert abs(float(sum(v[2] for v in z)-best)-x['gap_to_exact_ms'])<1e-9
 for actual,expected in zip(x['per_pending'],z):assert (actual['E'],actual['G'])==expected[:2] and abs(actual['cost_ms']-float(expected[2]))<1e-9
checks=0
for selected,value in allcost.items():
 for candidate in cs:
  if candidate in selected:continue
  newer=tuple(sorted(selected+(candidate,)));delta=float(value-allcost[newer])
  bm=r['benefit_matrix_ms']
  old=sum(max([bm[c][i] for c in selected]+[0]) for i in range(4))
  new=sum(max([bm[c][i] for c in newer]+[0]) for i in range(4))
  assert abs(delta-(new-old))<1e-9 and delta>=-1e-9
  checks+=1
for k in range(1,5):
 e=next(x for x in ex['exact'] if x['K']==k)
 best=min(v for s,v in allcost.items() if len(s)<=k)
 ties=[list(s) for s,v in allcost.items() if len(s)<=k and abs(v-best)<=Decimal('1e-9')]
 assert ties==e['all_optimal_sets']
for path,h in json.loads((P/'input_integrity.json').read_text())['sources_sha256'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h
out={'status':'PASS','decimal_subsets':128,'policy_results_checked':24,'marginal_transitions_checked':checks,'exact_budgets_checked':4,'frozen_inputs_unchanged':True,'说明':'从原始严格边直接构造前沿，以独立Decimal公式复核；不调用任何选择器'}
(P/'validation.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(out)
