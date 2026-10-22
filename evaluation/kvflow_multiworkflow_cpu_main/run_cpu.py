"""在冻结情景候选域复用五种既有选择规则，统一评价当前与事后目标。"""
import csv,json,sys,time,hashlib,itertools
from pathlib import Path
from flowstate.optimizer import GlobalOptimizer
from flowstate.state_catalog import CheckpointCandidate
from evaluation.kvflow_inspired_retention_cpu.policy import select,compute_ste
from evaluation.sota_policies import MarconiStylePolicy,_build_marconi_metrics
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
sys.path.insert(0,str(ROOT/'evaluation/p1_strong_baseline_validation'))
import common,wfc
NAMES=['LRU','Marconi Adaptation','KVFlow-inspired','WFC','FlowState-Greedy']
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
 with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def table(name,rows):
 with (OUT/name).open('x') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader()
  for r in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list,tuple)) else v for k,v in r.items()})
def verify_lock():
 for p,h in read(OUT/'input_lock.json')['sha256'].items():
  if sha(Path(p))!=h:raise ValueError('冻结输入或策略源码摘要变化：'+p)
def epoch(case,k,scope='current'):
 calls=case['calls'];order=case['predeclared_evaluation_order']
 qs=[next(p for p in calls if p['continuation_id']==i) for i in order]
 qs=[p for p in qs if scope=='all' or (p['ready'] if scope=='current' else not p['ready'])]
 ids={p['continuation_id'] for p in qs}
 return common.Epoch(tuple(common.Candidate(c['checkpoint_id'],c['token_position_L'],tuple(x for x in c['token_compatible_calls'] if x in ids)) for c in case['candidates']),tuple(common.Pending(p['continuation_id'],p['planning_target_T']) for p in qs),k)
def solve(case,k):
 ep=epoch(case,k);cs=case['candidates'];core=tuple(CheckpointCandidate(c['checkpoint_id'],c['checkpoint_id'],(),c['token_position_L'],1,True) for c in cs)
 # 此处True仅供既有CPU选择内核表示情景可选性，不是实际驻留认证。
 greedy=GlobalOptimizer(common.MODEL)._select_candidates(core,k,lambda chosen:common.score(ep,tuple(c.checkpoint_id for c in chosen))['cost_ms'])
 gr=tuple(c.checkpoint_id for c in greedy.selected)
 kv,priority=select(cs,case['graph'],case['declarations'],case['epoch_id'],case['configuration_digest'],k)
 wf,wt=wfc.select(ep);rec={c['checkpoint_id']:c['last_access_order'] for c in cs};flop={c['checkpoint_id']:c['token_position_L'] for c in cs}
 mar=MarconiStylePolicy().select(core,k,rec,flop,1.0).selected_checkpoint_ids
 choices={'LRU':tuple(sorted(rec,key=lambda x:(-rec[x],x))[:k]),'Marconi Adaptation':mar,'KVFlow-inspired':kv,'WFC':wf,'FlowState-Greedy':gr}
 gt=[]
 for i,chosen in enumerate(gr):
  prefix=gr[:i];before=common.score(ep,prefix)['cost_ms']
  gt.append({'prefix':prefix,'chosen':chosen,'candidate_marginal_ms':{c.identifier:before-common.score(ep,prefix+(c.identifier,))['cost_ms'] for c in ep.candidates if c.identifier not in prefix}})
 return choices,{'KVFlow-inspired':priority,'WFC':wt,'FlowState-Greedy':gt,'LRU':rec,'Marconi Adaptation':_build_marconi_metrics(core,rec,flop,1.0)}
def main():
 verify_lock();assert not (OUT/'cpu_results.json').exists(),'禁止覆盖或根据结果重跑'
 write(OUT/'execution_lock.json',{'started_unix_ns':time.time_ns(),'inputs_sha256':sha(OUT/'input_lock.json'),'implementation_sha256':sha(Path(__file__))})
 cases=read(OUT/'workloads.json')['cases'];rows=[];pairs=[];priorities=[];traces=[];ledger=[];support=[]
 cal=list(csv.DictReader((ROOT/'evaluation/runtime_artifacts/recovery_model_freeze_20260826_154235_266020/calibration_points.csv').open()));min_t=min(int(x['target_tokens']) for x in cal);max_t=max(int(x['target_tokens']) for x in cal);max_g=max(int(x['gap_tokens']) for x in cal)
 for case in cases:
  ep=epoch(case,len(case['candidates']));required=[]
  for p in ep.pending:
   best=max(c.position for c in ep.candidates if p.identifier in c.compatible);matches=[c.identifier for c in ep.candidates if p.identifier in c.compatible and c.position==best]
   assert len(matches)==1;required+=matches
  kmin=len(set(required))
  for c in case['candidates']:
   support.append({'structure':case['structure'],'candidate':c['checkpoint_id'],'L':c['token_position_L'],'parent':c['parent_candidate'],'current_support':[p.identifier for p in ep.pending if p.identifier in c['token_compatible_calls']],'all_declared_support':c['token_compatible_calls'],'state_status':'PROPOSED','actual_runtime_mapping':'UNKNOWN','logical_cost':1,'common_protected':False,'expected_slot_count':1,'last_access_order':c['last_access_order'],'creation_order':c['creation_order'],'provenance':'合成声明，未执行Native访问'})
  for k in read(OUT/'protocol.json')['budgets']:
   choices,trace=solve(case,k);traces.append({'structure':case['structure'],'K':k,'trace':trace})
   kr={x['checkpoint_id']:x for x in trace['KVFlow-inspired']}
   for c in case['candidates']:
    cid=c['checkpoint_id'];priorities.append({'structure':case['structure'],'K':k,'candidate':cid,'L':c['token_position_L'],'KVFlow_STE':kr[cid]['agent_ste'],'KVFlow_priority':kr[cid]['priority'],'LRU_access_order':trace['LRU'][cid],'Marconi_utility':trace['Marconi Adaptation'][cid],'Marconi_proxy_tokens':c['token_position_L'],'selected_by':[name for name,s in choices.items() if cid in s],'dynamic_priorities':'WFC/Greedy每轮候选排名与边际见selection_traces.json'})
   condition={}
   for name,s in choices.items():
    now=common.score(epoch(case,k),s);future=common.score(epoch(case,k,'future'),s);all_score=common.score(epoch(case,k,'all'),s)
    assert abs(now['cost_ms']+future['cost_ms']-all_score['cost_ms'])<1e-8
    for score in [now,future,all_score]:
     for p in score['per_pending']:
      p['calibration_status']='EXACT_ZERO_BY_DEFINITION' if p['G']==0 else ('OUTSIDE_CALIBRATION_TARGET_RANGE' if not min_t<=p['T']<=max_t else 'INSIDE_CALIBRATION_BOUNDS_NOT_POINTWISE_CERTIFIED')
      assert p['G']<=max_g and p['T']<=common.MODEL.metadata.maximum_target_tokens
    condition[name]=now
    rows.append({'structure':case['structure'],'epoch':case['epoch_id'],'K':k,'candidate_count':len(case['candidates']),'current_pending_count':len(ep.pending),'future_declared_count':len(case['calls'])-len(ep.pending),'K_min_frontier':kmin,'policy':name,'selected':s,'selected_count':len(s),'expected_managed_resident_slots':len(s),'uncontrolled_common_slots':0,'expected_evicted_count':len(case['candidates'])-len(s),'current_J_ms':now['cost_ms'],'current_E_G':now['per_pending'],'future_static_J_ms':future['cost_ms'],'future_E_G':future['per_pending'],'all_calls_static_J_ms':all_score['cost_ms'],'all_calls_E_G':all_score['per_pending'],'state_status':'CPU_PROPOSED','cost_status':'模型数值；非GPU延迟，非零项超出校准目标位置范围'})
    ledger.append({'structure':case['structure'],'K':k,'policy':name,'kept':s,'eviction_complement':[c['checkpoint_id'] for c in case['candidates'] if c['checkpoint_id'] not in s],'common_retained':[],'expected_slots_after':len(s),'actual_slots_after':'UNKNOWN','recurrent_pool_assumed_capacity':28,'planned_initial_slots':len(case['candidates']),'planned_headroom_before':28-len(case['candidates']),'runtime_requirement':'必须确认完整候选域、别名、引用、槽位及安全驱逐；出现额外可恢复状态则停止，不能隐式保护或删除'})
   for a,b in itertools.combinations(NAMES,2):
    x,y=condition[a],condition[b];same=set(choices[a])==set(choices[b]);front=common.frontier_vector(x)==common.frontier_vector(y);delta=x['cost_ms']-y['cost_ms']
    pairs.append({'structure':case['structure'],'K':k,'policy_a':a,'policy_b':b,'relation':'SAME_SET_SAME_FRONTIER' if same and front else ('DIFFERENT_SET_SAME_FRONTIER' if front else 'DIFFERENT_FRONTIER'),'J_a_ms':x['cost_ms'],'J_b_ms':y['cost_ms'],'delta_a_minus_b_ms':delta,'result_a':'WIN' if delta< -1e-9 else 'LOSS' if delta>1e-9 else 'TIE'})
 table('cpu_comparison.csv',rows);table('pairwise_comparison.csv',pairs);table('candidate_priorities.csv',priorities);table('candidate_resource_ledger.csv',support)
 write(OUT/'retention_ledger.json',ledger);write(OUT/'selection_traces.json',traces);write(OUT/'cpu_results.json',{'rows':rows,'pairs':pairs,'calibration_bounds':{'min_T':min_t,'max_T':max_t,'max_G':max_g},'scope':'三种预定义合成结构、六个预算条件、三十个策略行，不是六个独立工作负载'})
 verify_lock()
 print('完成30策略行、60成对比较，冻结输入摘要未变')
if __name__=='__main__':main()
