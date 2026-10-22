"""只读冻结输入，审计局部替换、排序结构及统一CPU求解成本。"""
import csv,hashlib,itertools,json,platform,statistics,sys,time
from pathlib import Path
from dataclasses import replace
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'evaluation/p1_strong_baseline_validation'))
import common,pcb,ldr,wfc
from flowstate.optimizer import GlobalOptimizer
from flowstate.state_catalog import CheckpointCandidate
EPS=1e-9
POLICIES=('Exact','Greedy','Greedy+1-Swap','WFC','StandaloneTopK','PCB','LDR')
SOURCES={}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):SOURCES[str(p)]=sha(p);return json.loads(Path(p).read_text())
def save(name,x):
 with (OUT/name).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2)
def csvout(name,rows):
 with (OUT/name).open('x') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader()
  for r in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list,tuple)) else v for k,v in r.items()})
def manifest(base):
 x=read(base/'evidence_manifest.json')
 for n,h in x['files'].items():
  p=Path(n) if Path(n).is_absolute() else base/n
  assert sha(p)==h,str(p);SOURCES[str(p)]=h

def load():
 e2=ROOT/'evaluation/independent_runtime_allocation_e2';c4=ROOT/'evaluation/multishare_allocation_gate_b5c4'
 manifest(e2);manifest(c4)
 lock=read(ROOT/'evaluation/strict_compatibility_b4c/input_integrity.json')['sha256']
 for n in ('common.py','pcb.py','ldr.py','wfc.py'):
  p=ROOT/'evaluation/p1_strong_baseline_validation'/n;assert sha(p)==lock[str(p)];SOURCES[str(p)]=sha(p)
 cases=[]
 for g in ('w1','w2','w3','w4'):
  s=read(e2/g/'strict_epoch_snapshot.json');old=read(e2/g/'allocation_results.json')
  assert len(s['edges'])==len(s['candidates'])*len(s['pendings']) and not any(e['status']=='UNKNOWN' for e in s['edges'])
  ce=common.Epoch(tuple(common.Candidate(c['checkpoint_id'],c['token_position_L'],tuple(e['continuation_id'] for e in s['edges'] if e['checkpoint_id']==c['checkpoint_id'] and e['status']=='EPOCH_ADMISSIBLE')) for c in s['candidates']),tuple(common.Pending(p['continuation_id'],p['planning_target_T']) for p in s['pendings']),4)
  cases.append((g,ce,old['rows']))
 x=read(c4/'results.json');z=x['input'];cases.append(('B5-C4',common.Epoch(tuple(common.Candidate(c['identifier'],c['position'],tuple(c['compatible'])) for c in z['candidates']),tuple(common.Pending(**p) for p in z['pending']),4),x['rows']))
 return cases

def swap(ep,selected,trace=False):
 """固定字典序首改善替换；严格下降保证有限终止，不跨等值平台。"""
 current=tuple(sorted(selected));ids=tuple(sorted(c.identifier for c in ep.candidates));steps=[];evaluations=0
 while True:
  before=common.score(ep,current)['cost_ms'];accepted=None
  for removed in current:
   for added in ids:
    if added in current:continue
    trial=tuple(sorted((set(current)-{removed})|{added}));after=common.score(ep,trial)['cost_ms'];evaluations+=1
    if before-after>EPS:
     accepted=trial;steps.append({'before_set':current,'remove':removed,'add':added,'after_set':trial,'before_J':before,'after_J':after,'gain_ms':before-after});break
   if accepted is not None:break
  if accepted is None:break
  current=accepted
 return current,{'swaps':len(steps),'neighbor_evaluations':evaluations,'steps':steps}

def solvers(ep):
 ids=tuple(sorted(c.identifier for c in ep.candidates));core=tuple(CheckpointCandidate(c.identifier,'严格矩阵占位',(),c.position,1,True) for c in ep.candidates);bm=common.benefit_matrix(ep)
 def greedy():return tuple(c.checkpoint_id for c in GlobalOptimizer(common.MODEL)._select_candidates(core,ep.k,lambda cs:common.score(ep,tuple(c.checkpoint_id for c in cs))['cost_ms']).selected)
 def exact():
  scored=[(common.score(ep,x)['cost_ms'],x) for k in range(ep.k+1) for x in itertools.combinations(ids,k)];best=min(v for v,_ in scored)
  return min((x for v,x in scored if abs(v-best)<=EPS),key=lambda x:(len(x),x))
 return {'Greedy':greedy,'Exact':exact,'Greedy+1-Swap':lambda:swap(ep,greedy())[0],'WFC':lambda:wfc.select(ep)[0],'PCB':lambda:pcb.select(ep)[0],'LDR':lambda:ldr.select(ep)[0],'StandaloneTopK':lambda:tuple(c for c in sorted(ids,key=lambda c:(-sum(bm[c]),c)) if sum(bm[c])>EPS)[:ep.k]}

def main():
 save('protocol_lock.json',{'protocol_sha256':sha(OUT/'protocol.json'),'script_sha256':sha(Path(__file__))})
 cases=load();rows=[];timings=[];traces=[];swaps=[]
 names={'FlowState-Greedy':'Greedy','FlowState':'Greedy','FlowState-Exact':'Exact','Exact OPT':'Exact'}
 for name,base,old in cases:
  for k in (1,2,3,4):
   ep=replace(base,k=k);func=solvers(ep);selected={p:func[p]() for p in POLICIES};opt=common.score(ep,selected['Exact'])['cost_ms']
   for r in old:
    if r['K']==k:
     p=names.get(r['policy'],r['policy']);assert set(selected[p])==set(r['selected']) and abs(common.score(ep,selected[p])['cost_ms']-r['J_ms'])<=EPS,(name,k,p)
   ss,detail=swap(ep,selected['Greedy']);assert ss==selected['Greedy+1-Swap'];swaps.append({'case':name,'K':k,**detail})
   current=common.score(ep,ss)['cost_ms']
   for a in ss:
    for b in sorted(set(c.identifier for c in ep.candidates)-set(ss)):
     assert common.score(ep,tuple((set(ss)-{a})|{b}))['cost_ms']>=current-EPS
   native=wfc.select(ep)[1]
   prefix=[]
   for i,c in enumerate(selected['Greedy']):
    step=native['steps'][i];assert tuple(step['prefix'])==tuple(prefix)
    cost=common.score(ep,prefix)['cost_ms'];rank=[]
    for z in step['ranked_candidates']:
     rank.append({**z,'gain_ms':cost-common.score(ep,tuple(prefix)+(z['checkpoint_id'],))['cost_ms']})
    traces.append({'case':name,'K':k,'step':i+1,'prefix':prefix[:],'greedy_chosen':c,'wfc_chosen':step['chosen'],'ranked':rank});prefix.append(c)
   expected={p:tuple(sorted(x)) for p,x in selected.items()}
   for repeat in range(-3,31):
    order=list(POLICIES);offset=repeat%len(order);order=order[offset:]+order[:offset]
    for order_index,p in enumerate(order):
     t=time.perf_counter_ns();out=func[p]();elapsed=time.perf_counter_ns()-t
     assert tuple(sorted(out))==expected[p]
     if repeat>=0:timings.append({'case':name,'K':k,'policy':p,'repeat':repeat,'order_index':order_index,'elapsed_ms':elapsed/1e6})
    t=time.perf_counter_ns();out,_=swap(ep,selected['Greedy']);elapsed=time.perf_counter_ns()-t;assert out==ss
    if repeat>=0:timings.append({'case':name,'K':k,'policy':'Swap-only','repeat':repeat,'order_index':7,'elapsed_ms':elapsed/1e6})
   for p in POLICIES:
    score=common.score(ep,selected[p]);tt=[r['elapsed_ms'] for r in timings if r['case']==name and r['K']==k and r['policy']==p]
    rows.append({'case':name,'K':k,'policy':p,'selected':selected[p],'J_ms':score['cost_ms'],'gap_exact_ms':score['cost_ms']-opt,'per_pending':score['per_pending'],'swaps':detail['swaps'] if p=='Greedy+1-Swap' else 0,'solve_mean_ms':statistics.mean(tt),'solve_median_ms':statistics.median(tt),'solve_min_ms':min(tt),'solve_max_ms':max(tt)})
 csvout('condition_results.csv',rows);csvout('timing_raw.csv',timings);save('selection_traces.json',traces);save('swap_traces.json',swaps)
 summary=[]
 for p in POLICIES+('Swap-only',):
  ts=[r['elapsed_ms'] for r in timings if r['policy']==p]
  summary.append({'policy':p,'runs':len(ts),'mean_ms':statistics.mean(ts),'median_ms':statistics.median(ts),'min_ms':min(ts),'max_ms':max(ts),'exact_conditions':sum(r['policy']==p and abs(r['gap_exact_ms'])<=EPS for r in rows)})
 csvout('timing_summary.csv',summary)
 save('results.json',{'gate':'PASS','rows':rows,'summary':summary,'swap_traces':swaps,'environment':{'python':sys.version,'platform':platform.platform(),'cpu':Path('/proc/cpuinfo').read_text().split('model name')[1].split('\n')[0],'clock':str(time.get_clock_info('perf_counter'))}})
 for p,h in SOURCES.items():assert sha(p)==h,p
 save('source_integrity.json',{'status':'PASS','sha256':SOURCES,'unchanged':True})
 print(json.dumps(summary,ensure_ascii=False,indent=2));print('替换',[(s['case'],s['K'],s['swaps']) for s in swaps])
if __name__=='__main__':main()
