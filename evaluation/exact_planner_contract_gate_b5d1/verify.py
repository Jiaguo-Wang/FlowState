"""独立Decimal穷举准入条件，并复核容差、规模和已记录计时。"""
import hashlib,itertools,json,math,statistics,sys
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
P=Path(__file__).resolve().parent;ROOT=P.parents[1];sys.path.insert(0,str(ROOT))
from evaluation.exact_planner_contract_gate_b5d1 import planner as a
from flowstate.strict_control import validate_plan
from dataclasses import replace
r=json.loads((P/'correctness_tests.json').read_text());oracle=[]
for row in r['conditions']:
 if row['solver']!='Exact':continue
 g=row['group'];k=row['K']
 path=ROOT/'evaluation/multishare_runtime_gate_b5c3/strict_epoch_snapshot.json' if g=='b5c3' else ROOT/f'evaluation/strict_compatibility_b4b1_output/capture_g001_g006_run01/{g}/snapshot.json'
 s=json.loads(path.read_text());cs={c['checkpoint_id']:c for c in s['candidates']};ps=s['pendings'];edges={(e['checkpoint_id'],e['continuation_id']) for e in s['edges'] if e['status']=='EPOCH_ADMISSIBLE'}
 def phi(g,t):
  g=Decimal(g)/1024;t=Decimal(t)/1024
  return Decimal('37.828150')*g+Decimal('0.345974143')*g*t-Decimal('0.156201917')*g*g
 scored=[]
 for n in range(k+1):
  for selected in itertools.combinations(sorted(cs),n):
   cost=Decimal(0)
   for p in ps:
    pos=max([cs[c]['token_position_L'] for c in selected if (c,p['continuation_id']) in edges]+[0]);cost+=phi(p['planning_target_T']-pos,p['planning_target_T'])
   scored.append((cost,selected))
 best=min(v for v,_ in scored);winner=min((x for v,x in scored if abs(v-best)<=Decimal('1e-9')),key=lambda x:(len(x),x))
 assert list(winner)==row['plan']['selected_ids'] and abs(float(best)-row['plan']['recovery_cost_ms'])<=1e-9
 oracle.append({'group':g,'K':k,'subsets':len(scored),'status':'PASS'})
# 这些微型成本列表只测试排序合同，不属于新工作负载或策略收益矩阵。
ties=[([(0.,('b',)),(0.,('a',))],('a',)), ([(0.,('b','c')),(0.,('z',))],('z',)), ([(0.,('b',)),(0.5e-9,('a',))],('a',)), ([(0.,('b',)),(2e-9,('a',))],('b',)), ([(0.,()),(0.,('a',))],())]
for values,expected in ties:assert a.choose_exact(values)==expected
limits=[]
for n,m,k,expected in [(12,4,4,True),(13,4,1,False),(7,5,1,False),(7,4,5,False),(7,4,0,True)]:
 assert a.allowed(SimpleNamespace(candidates=range(n),pendings=range(m),K=k))==expected
 limits.append({'n':n,'m':m,'K':k,'allowed':expected,'说明':'仅规模谓词测试，不作为合法Epoch分配'})
path=ROOT/'evaluation/strict_compatibility_b4b1_output/capture_g001_g006_run01/g001/snapshot.json';data=path.read_bytes();ev=a.FrozenEvidence(data,hashlib.sha256(data).hexdigest());ep=a.from_frozen(ev,2)
checks=0
def cancel():
 global checks
 checks+=1
 return checks>=3
try:a.plan(ep,ev,cancel=cancel)
except a.Cancelled:pass
else:raise AssertionError('中途取消未拒绝')
plan=a.plan(ep,ev).plan
for bad in [replace(plan,selected_count=999),replace(plan,recovery_cost_ms=float('nan')),replace(plan,epoch_digest='0'*64),replace(plan,logical_eviction_ids=())]:
 try:validate_plan(ep,bad)
 except (ValueError,RuntimeError):pass
 else:raise AssertionError('非法Plan未拒绝')
raw=json.loads((P/'raw_timing.json').read_text());summary=json.loads((P/'timing_summary.json').read_text());assert len(raw)==480
for row in summary:
 batch=[x for x in raw if x['group']==row['group'] and x['K']==row['K'] and x['solver']==row['solver']];assert len(batch)==20
 for field in ['prepare_ms','benefits_ms','solve_ms','plan_validation_ms','total_ms']:
  values=sorted(x[field] for x in batch)
  assert statistics.median(values)==row[field+'_median'] and values[18]==row[field+'_p95']
 for x in batch:assert abs(sum(x[f] for f in ['prepare_ms','benefits_ms','solve_ms','plan_validation_ms'])-x['total_ms'])<1e-8
for path,h in json.loads((P/'implementation_lock.json').read_text())['sources_sha256'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h
(P/'independent_validation.json').write_text(json.dumps({'status':'PASS','decimal_oracle':oracle,'tie_tests':len(ties),'limit_predicate_tests':limits,'mid_solve_cancel':'PASS','invalid_plan_tests':4,'timing_records':480,'timing_recalculation':'PASS','frozen_unchanged':True},ensure_ascii=False,indent=2)+'\n')
print('独立Oracle、同分、取消、计划与计时复核PASS')
