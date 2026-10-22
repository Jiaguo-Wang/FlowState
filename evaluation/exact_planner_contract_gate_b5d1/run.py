"""固定二十条件回归、负面合同和统一阶段CPU计时。"""
import copy,csv,hashlib,json,math,os,platform,statistics,sys,time
from dataclasses import asdict,replace
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from evaluation.exact_planner_contract_gate_b5d1 import planner as a
from flowstate.strict_epoch import _digest
from flowstate.optimizer import GlobalOptimizer
sources={}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):sources[str(p)]=sha(p);return json.loads(p.read_text())
def save(n,x):(OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def corrupt(obj,**changes):
 """仅用于负面测试的独立对象，故意绕过构造器以检查入口拒绝。"""
 v=copy.copy(obj)
 for k,x in changes.items():object.__setattr__(v,k,x)
 return v
def main():
 assert not (OUT/'raw_timing.json').exists(),'不覆盖既有测量'
 for directory in ['strict_compatibility_b4c','multishare_allocation_gate_b5c4','multishare_runtime_gate_b5c3','solver_feasibility_gate_b5d0']:
  base=ROOT/'evaluation'/directory;m=read(base/'evidence_manifest.json')
  for name,h in m['files'].items():
   p=Path(name) if Path(name).is_absolute() else base/name
   assert sha(p)==h;sources[str(p)]=h
 for name in ['optimizer.py','strict_epoch.py','strict_control.py','state_catalog.py','recovery_model.py']:
  p=ROOT/'flowstate'/name;sources[str(p)]=sha(p)
 b4=read(ROOT/'evaluation/strict_compatibility_b4c/results.json');c4=read(ROOT/'evaluation/multishare_allocation_gate_b5c4/results.json');cases=[];correct=[]
 for g in ['g001','g002','g003','g004','b5c3']:
  base=ROOT/'evaluation/multishare_runtime_gate_b5c3'
  p=base/'strict_epoch_snapshot.json' if g=='b5c3' else ROOT/f'evaluation/strict_compatibility_b4b1_output/capture_g001_g006_run01/{g}/snapshot.json'
  s=read(p);provenance=None
  ev=a.FrozenEvidence(p.read_bytes(),sha(p))
  if g=='b5c3':
   phys=read(base/'physical_state_census.json');origins=read(base/'candidate_provenance.json');history=read(base/'prepared_inputs.json')
   ev=replace(ev,physical_snapshot=(base/'physical_state_census.json').read_bytes(),physical_sha256=sha(base/'physical_state_census.json'));provenance=(phys,origins,history)
  refs=c4['rows'] if g=='b5c3' else next(x for x in b4['snapshots'] if x['group']==g)['rows']
  for k in range(1,5):
   ep=a.from_frozen(ev,k,provenance);out=a.plan(ep,ev,allow_fallback=True)
   reference=next(x for x in refs if x['K']==k and x['policy']==('Exact OPT' if a.allowed(ep) else 'FlowState'))
   assert set(out.plan.selected_ids)==set(reference['selected']) and abs(out.plan.recovery_cost_ms-reference['J_ms'])<=1e-9
   production=GlobalOptimizer(a.MODEL).plan_strict(ep)
   greedyref=next(x for x in refs if x['K']==k and x['policy']=='FlowState')
   assert list(production.selected_ids)==greedyref['selected'] and abs(production.recovery_cost_ms-greedyref['J_ms'])<=1e-9
   correct.append({'group':g,'K':k,'status':'PASS','solver':out.solver,'optimality_certified':out.optimality_certified,'plan':asdict(out.plan),'reference':reference['selected'],'production_greedy_regression':'PASS'})
   cases.append((g,ep,ev))
 save('correctness_tests.json',{'status':'PASS','conditions':correct,'exact_conditions':sum(x['solver']=='Exact' for x in correct),'fallback_conditions':sum(x['solver']=='Greedy' for x in correct)})
 tests=[];ep=cases[0][1];ev=cases[0][2]
 def reject(name,fn,expected=ValueError):
  try:fn()
  except expected as e:tests.append({'name':name,'status':'PASS','exception':type(e).__name__,'reason':str(e)})
  else:raise AssertionError(name+'未拒绝')
 def call(x,y=ev):return a.plan(x,y,allow_fallback=True)
 reject('UNKNOWN配对',lambda:call(corrupt(ep,edges=(replace(ep.edges[0],status='UNKNOWN'),)+ep.edges[1:])))
 reject('缺失配对',lambda:call(corrupt(ep,edges=ep.edges[:-1])))
 reject('重复配对',lambda:call(corrupt(ep,edges=ep.edges+(ep.edges[0],))))
 reject('重复候选身份',lambda:call(corrupt(ep,candidates=ep.candidates+(ep.candidates[0],))))
 reject('重复Pending身份',lambda:call(corrupt(ep,pendings=ep.pendings+(ep.pendings[0],))))
 for k in [-1,1.5,True]:reject('非法K'+str(k),lambda k=k:call(corrupt(ep,K=k)))
 reject('Epoch身份变化',lambda:call(corrupt(ep,epoch_id='其他边界')))
 reject('配置摘要变化',lambda:call(corrupt(ep,configuration_digest='0'*64)))
 reject('内容摘要变化',lambda:call(corrupt(ep,content_digest='0'*64)))
 reject('输入来源摘要变化',lambda:call(ep,replace(ev,expected_sha256='0'*64)))
 reject('状态陈旧',lambda:call(ep,replace(ev,stale=True)))
 reject('冒充当前Runtime身份',lambda:call(ep,replace(ev,scope='current_runtime')))
 c=ep.candidates[0]
 reject('位置非法',lambda:call(corrupt(ep,candidates=(replace(c,candidate=replace(c.candidate,token_pos=c.candidate.token_pos+1)),)+ep.candidates[1:])))
 reject('不同预算成本',lambda:call(corrupt(ep,candidates=(replace(c,candidate=replace(c.candidate,memory_bytes=c.candidate.memory_bytes+1)),)+ep.candidates[1:])))
 reject('Recurrent缺失',lambda:call(corrupt(ep,candidates=(replace(c,candidate=replace(c.candidate,recurrent_resident=False)),)+ep.candidates[1:])))
 reject('Token摘要冲突',lambda:call(corrupt(ep,candidates=(replace(c,token_digest='0'*64),)+ep.candidates[1:])))
 # 有效重签的独立故障输入仍须经过物理证据门禁。
 for name,mutate in [('Slot别名',lambda s:s['candidates'][1]['state_identity'].update(target_mamba_slots=s['candidates'][0]['state_identity']['target_mamba_slots'])),('Node别名',lambda s:s['candidates'][1]['state_identity'].update(node_id=s['candidates'][0]['state_identity']['node_id'])),('非零引用',lambda s:s['candidates'][0]['state_identity']['references'].update(target_mamba_lock_ref=1))]:
  raw=json.loads(ev.snapshot);mutate(raw);data=json.dumps(raw).encode();fake=a.FrozenEvidence(data,hashlib.sha256(data).hexdigest());bad=a.from_frozen(fake,1)
  reject(name,lambda bad=bad,fake=fake:call(bad,fake))
 prepared=a.prepare(ep,ev);matrix=a.benefits(prepared)
 for val in [float('nan'),float('inf'),-1.0]:
  bad=((val,)+matrix[0][1:],)+matrix[1:]
  reject('非法收益'+str(val),lambda bad=bad:a.solve(prepared,bad,'Exact'))
 reject('取消不返回部分计划',lambda:a.plan(ep,ev,cancel=lambda:True),a.Cancelled)
 for g,e,v in cases:
  if not a.allowed(e):reject('规模拒绝'+g+str(e.K),lambda e=e,v=v:a.plan(e,v),a.LimitError)
 larger=next(x for x in cases if x[0]=='g002')
 reject('超规模UNKNOWN不可降级',lambda:a.plan(corrupt(larger[1],edges=(replace(larger[1].edges[0],status='UNKNOWN'),)+larger[1].edges[1:]),larger[2],allow_fallback=True))
 # 零预算合法；K超过候选数不是生产合同非法，按本轮规模规则拒绝Exact。
 zero=a.from_frozen(ev,0);z=a.plan(zero,ev);assert not z.plan.selected_ids
 for k in [5,13]:
  e=a.from_frozen(ev,k);reject('合法K超限'+str(k),lambda e=e:a.plan(e,ev),a.LimitError)
 save('negative_tests.json',{'status':'PASS','rejections':tests,'zero_budget':'PASS','说明':'故障注入只用内存独立副本，不改变任何冻结数据'})
 eligible=[x for x in cases if a.allowed(x[1])];timings=[]
 affinity=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,{affinity[0]})
 save('timing_protocol.json',{'warmup':2,'repetitions':20,'eligible_conditions':12,'order':'Exact和Greedy每轮交替先后，条件固定轮转','clock':'perf_counter_ns','scope':'已有StrictEpoch到完整Plan；共同prepare含validate、原始证据解析和物理身份检查；共同benefits；各自solve和finish含生产validate_plan','p95':'本地20重复最近秩第19项，不是在线P95','excluded':'初次原文件读取、from_frozen构造、SHA清单遍历、结果核对、写盘；prepare内部证据摘要检查仍计时'})
 save('environment.json',{'python':sys.version,'platform':platform.platform(),'cpuinfo':Path('/proc/cpuinfo').read_text().split('\n\n')[0],'initial_affinity':affinity,'affinity':sorted(os.sched_getaffinity(0)),'load_before':os.getloadavg()})
 for rep in range(-2,20):
  shift=(rep+2)%len(eligible)
  for g,e,v in eligible[shift:]+eligible[:shift]:
   start=time.perf_counter_ns();pr=a.prepare(e,v);t1=time.perf_counter_ns();bm=a.benefits(pr);t2=time.perf_counter_ns()
   shared=(t2-start)/1e6
   for solver in (['Exact','Greedy'] if rep%2==0 else ['Greedy','Exact']):
    t3=time.perf_counter_ns();chosen=a.solve(pr,bm,solver);t4=time.perf_counter_ns();plan=a.finish(pr,chosen);t5=time.perf_counter_ns()
    expected=next(x for x in correct if x['group']==g and x['K']==e.K)
    if solver=='Exact':assert list(plan.selected_ids)==expected['plan']['selected_ids'] or tuple(plan.selected_ids)==tuple(expected['plan']['selected_ids'])
    if rep>=0:timings.append({'group':g,'K':e.K,'solver':solver,'repetition':rep+1,'prepare_ms':(t1-start)/1e6,'benefits_ms':(t2-t1)/1e6,'solve_ms':(t4-t3)/1e6,'plan_validation_ms':(t5-t4)/1e6,'total_ms':shared+(t5-t3)/1e6})
  if rep in [-1,9,19]:print('计时轮次',rep+1,flush=True)
 save('raw_timing.json',timings)
 summary=[]
 for g,e,_ in eligible:
  for solver in ['Exact','Greedy']:
   batch=[x for x in timings if x['group']==g and x['K']==e.K and x['solver']==solver];row={'group':g,'K':e.K,'n':len(e.candidates),'m':len(e.pendings),'subsets':sum(math.comb(len(e.candidates),i) for i in range(e.K+1)),'solver':solver,'repetitions':20}
   for f in ['prepare_ms','benefits_ms','solve_ms','plan_validation_ms','total_ms']:
    values=sorted(x[f] for x in batch)
    for name,value in [('mean',statistics.mean(values)),('median',statistics.median(values)),('p95',values[18]),('min',values[0]),('max',values[-1])]:row[f+'_'+name]=value
   summary.append(row)
 with (OUT/'solver_timing_comparison.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=summary[0]);w.writeheader();w.writerows(summary)
 save('timing_summary.json',summary)
 assert all(sha(Path(p))==h for p,h in sources.items())
 save('implementation_lock.json',{'sources_sha256':sources,'new_code_sha256':{p.name:sha(p) for p in OUT.glob('*.py')},'frozen_unchanged':True})
 print('合同与计时完成。')
if __name__=='__main__':main()
