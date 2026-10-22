"""对固定五个冻结快照有限计时，复用历史直接枚举定义，不启动Runtime。"""
import csv,gc,hashlib,itertools,json,math,os,platform,statistics,sys,time
from dataclasses import replace
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'evaluation/p1_strong_baseline_validation'))
import common
B4=ROOT/'evaluation/strict_compatibility_b4c';C4=ROOT/'evaluation/multishare_allocation_gate_b5c4';C3=ROOT/'evaluation/multishare_runtime_gate_b5c3'
EPS=1e-9;sources={}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):sources[str(p)]=sha(p);return json.loads(p.read_text())
def save(n,v):(OUT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def prepare(text,k):
 """输入阶段含反序列化、完整准入域检查和不可变评分对象，不重建Token。"""
 s=json.loads(text);cs=s['candidates'];ps=s['pendings'];edges=s['edges']
 ci={c['checkpoint_id'] for c in cs};pi={p['continuation_id'] for p in ps}
 assert len(ci)==len(cs) and len(pi)==len(ps)
 assert {(e['checkpoint_id'],e['continuation_id']) for e in edges}==set(itertools.product(ci,pi)) and len(edges)==len(ci)*len(pi)
 for e in edges:
  assert e['status'] in ('EPOCH_ADMISSIBLE','REJECTED')
  if e['status']=='EPOCH_ADMISSIBLE':assert all(v is True for v in e['conditions'].values())
 return common.Epoch(tuple(common.Candidate(c['checkpoint_id'],c['token_position_L'],tuple(e['continuation_id'] for e in edges if e['checkpoint_id']==c['checkpoint_id'] and e['status']=='EPOCH_ADMISSIBLE')) for c in cs),tuple(common.Pending(p['continuation_id'],p['planning_target_T']) for p in ps),k)
def exact(epoch):
 """复用B4-C的完整子集、common.score及同分代表规则，仅限当前K。"""
 ids=tuple(sorted(c.identifier for c in epoch.candidates))
 subsets=[s for n in range(epoch.k+1) for s in itertools.combinations(ids,n)]
 scores={s:common.score(epoch,s) for s in subsets}
 best=min(scores[s]['cost_ms'] for s in subsets)
 ties=[s for s in subsets if abs(scores[s]['cost_ms']-best)<=EPS]
 chosen=min(ties,key=lambda s:(len(s),s))
 return chosen,best,len(subsets)
def summary(v):
 x=sorted(v);return dict(mean=statistics.mean(x),median=statistics.median(x),p95=x[math.ceil(.95*len(x))-1],minimum=x[0],maximum=x[-1])
def main():
 assert not (OUT/'raw_timings.json').exists(),'不覆盖已有计时'
 for base in (B4,C4,C3,ROOT/'evaluation/structural_exactness_gate_b5c0'):
  m=read(base/'evidence_manifest.json')
  for name,h in m['files'].items():
   p=Path(name) if Path(name).is_absolute() else base/name
   assert sha(p)==h,str(p);sources[str(p)]=h
 lock=read(B4/'input_integrity.json')['sha256']
 for p in (ROOT/'evaluation/p1_strong_baseline_validation/common.py',ROOT/'flowstate/recovery_model.py'):
  assert sha(p)==lock[str(p)];sources[str(p)]=sha(p)
 old=read(B4/'results.json');new=read(C4/'results.json');cases=[];structures=[]
 for g in ['g001','g002','g003','g004','b5c3']:
  path=C3/'strict_epoch_snapshot.json' if g=='b5c3' else ROOT/f'evaluation/strict_compatibility_b4b1_output/capture_g001_g006_run01/{g}/snapshot.json'
  raw=read(path);text=path.read_text();ep=prepare(text,4);bm=common.benefit_matrix(ep)
  refs=new['rows'] if g=='b5c3' else next(s for s in old['snapshots'] if s['group']==g)['rows']
  pairs=[]
  for c in raw['candidates']:
   for p in raw['pendings']:
    e=next(e for e in raw['edges'] if e['checkpoint_id']==c['checkpoint_id'] and e['continuation_id']==p['continuation_id'])
    pure=c['token_position_L']<=p['planning_target_T'] and c['token_ids']==p['token_ids'][:c['token_position_L']]
    pairs.append({'candidate':c['checkpoint_id'],'pending':p['continuation_id'],'pure_prefix_position':pure,'admissible':e['status']=='EPOCH_ADMISSIBLE','extra_filter_difference':pure!=(e['status']=='EPOCH_ADMISSIBLE')})
  supports=[{i for i,v in enumerate(values) if v>0} for values in bm.values()]
  states=[c['state_identity'] for c in raw['candidates']];slots=[slot for s in states for slot in s['target_mamba_slots']]
  laminar=all(not(a&b) or a<=b or b<=a for a,b in itertools.combinations(supports,2))
  structures.append({'group':g,'n':len(ep.candidates),'m':len(ep.pending),'pairs':pairs,'extra_filter_differences':sum(p['extra_filter_difference'] for p in pairs),'laminar':laminar,'shared_candidates':sum(len(s)>1 for s in supports),'unique_shared_supports':len({tuple(sorted(s)) for s in supports if len(s)>1}),'unique_slots':len(set(slots))==len(slots),'one_slot_each':all(len(s['target_mamba_slots'])==1 for s in states)})
  j0=common.score(ep,())['cost_ms']
  for n in range(5):
   for chosen in itertools.combinations(tuple(bm),n):
    f=sum(max([bm[c][i] for c in chosen]+[0]) for i in range(len(ep.pending)))
    assert abs(j0-f-common.score(ep,chosen)['cost_ms'])<=EPS
  for k in range(1,5):
   reference=next(r for r in refs if r['K']==k and r['policy']=='Exact OPT')
   cases.append((g,k,text,reference))
 protocol={'groups':[s['group'] for s in structures],'budgets':[1,2,3,4],'warmup':3,'repetitions':30,'order':'每轮按固定条件表循环左移一项；无结果驱动筛选','timing':'perf_counter_ns；准备=内存JSON解析与合同域检查，收益=矩阵构建，求解=本预算全部子集生成/评分/同分选集；总时间为三个连续区间','excluded':'磁盘读取、文件SHA、一次性Token结构审计、结果核对及写盘不计时','search':'不缓存跨预算或跨重复分数，不剪枝；不删除支配候选','p95':'30次本地重复的最近秩第29项；不是在线P95'}
 save('measurement_protocol.json',protocol)
 affinity=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,{affinity[0]})
 save('environment.json',{'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'cpuinfo':Path('/proc/cpuinfo').read_text().split('\n\n')[0],'initial_affinity':affinity,'benchmark_affinity':sorted(os.sched_getaffinity(0)),'gc_enabled':gc.isenabled(),'load_before':os.getloadavg(),'timer_resolution':time.get_clock_info('perf_counter').resolution,'说明':'只限定本进程CPU亲和性，不改变其他任务；无频率或后台负载隔离'})
 timings=[]
 for repetition in range(-3,30):
  shift=(repetition+3)%len(cases)
  for g,k,text,ref in cases[shift:]+cases[:shift]:
   t0=time.perf_counter_ns();ep=prepare(text,k);t1=time.perf_counter_ns();bm=common.benefit_matrix(ep);t2=time.perf_counter_ns();chosen,value,count=exact(ep);t3=time.perf_counter_ns()
   assert abs(value-ref['J_ms'])<=EPS and set(chosen)==set(ref['selected']),(g,k)
   if repetition>=0:timings.append({'group':g,'K':k,'n':len(ep.candidates),'m':len(ep.pending),'subsets':count,'repetition':repetition+1,'prepare_ms':(t1-t0)/1e6,'benefit_ms':(t2-t1)/1e6,'solve_ms':(t3-t2)/1e6,'total_ms':(t3-t0)/1e6,'selected':chosen,'J_ms':value})
  if repetition in (-1,9,19,29):print('已完成轮次',repetition+1,flush=True)
 rows=[]
 for g,k,_,_ in cases:
  batch=[x for x in timings if x['group']==g and x['K']==k]
  row={key:batch[0][key] for key in ('group','K','n','m','subsets')};row['repetitions']=30
  for field in ('prepare_ms','benefit_ms','solve_ms','total_ms'):
   row.update({field+'_'+stat:val for stat,val in summary([x[field] for x in batch]).items()})
  rows.append(row)
 save('raw_timings.json',timings);save('timing_summary.json',rows);save('structure_audit.json',structures)
 with (OUT/'timing_summary.csv').open('w') as f:
  writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
 assert all(sha(Path(p))==h for p,h in sources.items())
 save('integrity.json',{'status':'PASS','sources':sources,'unchanged':True,'matched_frozen_conditions':20,'measured_solves':len(timings),'warmup_solves':60,'load_after':os.getloadavg()})
 print('有限计时结束，冻结结果20/20一致。')
if __name__=='__main__':main()
