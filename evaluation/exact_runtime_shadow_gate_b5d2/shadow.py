"""对新观察的不可变Epoch只做CPU规划和独立Oracle，不访问Runtime。"""
import csv,hashlib,itertools,json,time
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path
from evaluation.exact_planner_contract_gate_b5d1 import planner as a
OUT=Path('/b5out')
def blob(x):return json.dumps(x,ensure_ascii=False,sort_keys=True).encode()
def write(name,x):(OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def run(observed,states,inputs,observation_ms):
 started=time.perf_counter_ns();s=observed['strict_epoch'];data=blob(s);physical=blob(observed)
 write('live_epoch_snapshot.json',s)
 origins=[]
 for c in s['candidates']:
  seen=[h['request_id'] for h,state in zip(inputs['history'],states) if any(n['node_id']==c['state_identity']['node_id'] and n['token_ids']==c['token_ids'] and n['slots'] for n in state['before']['nodes'])]
  if not seen:raise ValueError('候选没有当前生命周期逐请求来源证据')
  origins.append({'checkpoint_id':c['checkpoint_id'],'first_observed_after':seen[0],'说明':'首次Census不是精确insert事件，不用作兼容判断'})
 evidence=a.FrozenEvidence(data,hashlib.sha256(data).hexdigest(),physical_snapshot=physical,physical_sha256=hashlib.sha256(physical).hexdigest())
 epoch=a.from_frozen(evidence,2,(observed,origins,inputs));prepared=a.prepare(epoch,evidence);t1=time.perf_counter_ns();matrix=a.benefits(prepared);t2=time.perf_counter_ns()
 if not a.allowed(epoch):raise a.LimitError('当前合法Epoch超出D1规模，明确拒绝Exact，不扩大范围')
 plans=[];timing=[{'stage':'runtime_observation_freeze_rpc','ms':observation_ms,'solver':'共同','说明':'包含Scheduler观察、GPU同步摘要、返回传输与客户端解码'} ,{'stage':'epoch_construction_authentication','ms':(t1-started)/1e6,'solver':'共同','说明':'CPU窗口外，含序列化、来源整理、原Strict合同与物理证据认证及快照写出'}, {'stage':'benefit_matrix','ms':(t2-t1)/1e6,'solver':'共同','说明':'仅一次共同预处理'}]
 for solver in ['Greedy','Exact']:
  t=time.perf_counter_ns();selected=a.solve(prepared,matrix,solver);u=time.perf_counter_ns();plan=a.finish(prepared,selected);v=time.perf_counter_ns()
  timing += [{'stage':'solve','ms':(u-t)/1e6,'solver':solver,'说明':'同一冻结Epoch和共同评分'}, {'stage':'plan_construction_validation','ms':(v-u)/1e6,'solver':solver,'说明':'调用既有validate_plan'}]
  plans.append({'solver':solver,'optimality_certified':solver=='Exact','validation':'PASS','plan':asdict(plan)})
 # Oracle不读取历史目标；独立Decimal公式及当前边穷举。
 t=time.perf_counter_ns();candidates={c['checkpoint_id']:c for c in s['candidates']};pending=s['pendings'];edges={(e['checkpoint_id'],e['continuation_id']) for e in s['edges'] if e['status']=='EPOCH_ADMISSIBLE'}
 def phi(g,t):
  g=Decimal(g)/1024;t=Decimal(t)/1024
  return Decimal('37.828150')*g+Decimal('0.345974143')*g*t-Decimal('0.156201917')*g*g
 scored=[]
 for k in range(3):
  for selected in itertools.combinations(sorted(candidates),k):
   j=sum(phi(p['planning_target_T']-max([candidates[c]['token_position_L'] for c in selected if (c,p['continuation_id']) in edges]+[0]),p['planning_target_T']) for p in pending)
   scored.append((j,selected))
 best=min(v for v,_ in scored);chosen=min((c for v,c in scored if abs(v-best)<=Decimal('1e-9')),key=lambda c:(len(c),c))
 exact=plans[1]['plan'];assert tuple(exact['selected_ids'])==chosen and abs(exact['recovery_cost_ms']-float(best))<=1e-9
 timing.append({'stage':'independent_oracle','ms':(time.perf_counter_ns()-t)/1e6,'solver':'审计','说明':'不计入任一规划器独立路径'})
 write('current_epoch_oracle.json',{'status':'PASS','selected':chosen,'J_ms':float(best),'subsets':len(scored)})
 write('candidate_provenance.json',origins)
 write('shadow_plan_comparison.json',{'status':'PASS','epoch':asdict(epoch),'logical_candidate_ids':prepared.ids,'plans':plans,'oracle':'PASS','physical_binding':None,'freshness':'只对本次已冻结Epoch有效；不保证未来应用时新鲜'})
 with (OUT/'shadow_timing.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=['stage','ms','solver','说明']);w.writeheader();w.writerows(timing)
 return plans
