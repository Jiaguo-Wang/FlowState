"""只读汇总既有快照、执行记录与冻结比较，不构造候选或运行选择器。"""
import csv,hashlib,itertools,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
hashes={}
def read(path):
    p=Path(path);p=p if p.is_absolute() else ROOT/p
    b=p.read_bytes();hashes[str(p)]=hashlib.sha256(b).hexdigest()
    return json.loads(b)
def lines(path):
    p=ROOT/path;b=p.read_bytes();hashes[str(p)]=hashlib.sha256(b).hexdigest();return [json.loads(s) for s in b.splitlines() if s]
def save(name,x):(OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def csvsave(name,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
rows=lines('evaluation/p1_strong_baseline_validation/per_case_results.jsonl');assert len(rows)==573
inventory=[];comparisons=[];by={};counts={};correctness=[]
for lineno,r in enumerate(rows,1):
    inp=r['input'];candidates=inp['candidates'];bm=r['benefit_matrix_ms'];supports={c:{i for i,x in enumerate(v) if x>0} for c,v in bm.items()}
    shared=sum(len(v)>1 for v in supports.values());overlap=sum(bool(a&b) for a,b in itertools.combinations(supports.values(),2));pop=r['population'];sid=r['snapshot_id']
    key=(pop,sid);entry=by.setdefault(key,dict(dataset=pop,snapshot_id=sid,continuations=len(inp['pending']),candidates=len(candidates),positive_shared_candidates=shared,positive_shared_edges=sum(len(v) for v in supports.values() if len(v)>1),overlap_pairs=overlap,K_min=r['budget_structure']['minimum_checkpoints_for_all_frontiers'],budgets=[],source=str(ROOT/'evaluation/p1_strong_baseline_validation/per_case_results.jsonl')+f':{lineno}',matrix_scope='冻结历史矩阵，非本轮重新认证',pending_evidence='受控回放 barrier' if pop=='OpenHands' else '派生 current-context，非原生 Ready',executable_shared_evidence='未在正式矩阵中出现跨请求正收益' if pop=='OpenHands' else '正式新增边有 token 谓词冲突'))
    entry['budgets'].append(r['k'])
    for policy in ['PCB','LDR','WFC','StandaloneTopK']:
        delta=r['policies'][policy]['cost_ms']-r['policies']['FlowState']['cost_ms']
        comparisons.append(dict(dataset=pop,snapshot_id=sid,K=r['k'],policy=policy,baseline_minus_flowstate_ms=delta,outcome='W' if delta>1e-9 else 'L' if delta < -1e-9 else 'T',source=entry['source'],evidence_scope=entry['matrix_scope']))
for key,e in by.items():
    e['budgets']=json.dumps(e['budgets']);inventory.append(e)
    if key[0]=='OpenHands':
        ordinal=int(key[1].split('-g')[1].split('-')[0]);p=f'evaluation/runtime_artifacts/rq3_openhands_main_formal_20260904_001017/runtime_correctness/g{ordinal:03d}.json';r=read(p)
        correctness.append(dict(snapshot_id=key[1],source=p,abort_reason=r['abort_reason'],completed_requests=len(r['request_rows']),requests_complete=all(x['request_completed'] for x in r['request_rows']),boundary_no_future=all(not x['pending_turn_output_read'] and not x['r_plus_2_message_consumed'] and not x['future_checkpoint_read'] for x in r['boundary_audit']),final_mamba_count=r['census_rows'][-1]['mamba_node_count']))
for pop in ['OpenHands','AgentX']:
    rr=[r for r in rows if r['population']==pop];ee=[e for e in inventory if e['dataset']==pop]
    counts[pop]=dict(snapshots=len(ee),conditions=len(rr),continuation_range=[min(e['continuations'] for e in ee),max(e['continuations'] for e in ee)],candidate_range=[min(e['candidates'] for e in ee),max(e['candidates'] for e in ee)],K_min_range=[min(e['K_min'] for e in ee),max(e['K_min'] for e in ee)],tight_conditions=sum(r['k']<r['budget_structure']['minimum_checkpoints_for_all_frontiers'] for r in rr),shared_snapshots=sum(e['positive_shared_candidates']>0 for e in ee),comparisons={p:{v:sum(c['dataset']==pop and c['policy']==p and c['outcome']==v for c in comparisons) for v in ['W','T','L']} for p in ['PCB','LDR','WFC','StandaloneTopK']})
exact=[]
for pop,path in [('OpenHands','evaluation/runtime_artifacts/rq3_formal_policy_eval_20260904_110011/per_snapshot_results.jsonl'),('AgentX','rq3_agentx_policy_eval_20260906_214956/per_case_results.jsonl')]:
    for i,r in enumerate(lines(path),1):
        e=r.get('exact_opt')
        if e and e.get('tractable'):
            exact.append(dict(dataset=pop,snapshot_id=r['snapshot_id'],K=r['k'],gap_ms=e['flowstate_vs_exact']['absolute_cost_gap_ms'],source=path+f':{i}'))
    counts[pop]['exact_cases']=sum(r['dataset']==pop for r in exact)
    counts[pop]['exact_matches']=sum(r['dataset']==pop and abs(r['gap_ms'])<=1e-9 for r in exact)
lm=lines('agentic_workflow_dataset_research/phase2_submission_inputs/runtime_candidate_manifest.jsonl')
for r in lm:
    ids=[]
    for p,h in zip(r['input_artifact_paths'],r['input_sha256']):
        d=read(p);assert hashes[p]==h;ids.append(d['input_ids'])
    a,b=ids;lcp=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),min(len(a),len(b)));assert lcp==r['prefix_tokens']
    inventory.append(dict(dataset='LMCache',snapshot_id=r['case_id'],continuations=2,candidates='UNKNOWN：无同边界自然候选全集',positive_shared_candidates='内容前缀存在；候选数未认证',positive_shared_edges='UNKNOWN',overlap_pairs='UNKNOWN',K_min='UNKNOWN',budgets='未定义物理或历史分配预算',source=r['input_artifact_paths'],matrix_scope='TOKEN_COMPATIBLE 输入；不是 Pending 快照',pending_evidence='UNKNOWN',executable_shared_evidence='A5 C0 两条恢复路径已认证' if r['case_id']=='lmcache_10' else '未认证',exact_lcp=lcp))
a5=read('evaluation/submission_p0_lmcache_recovery/runtime_evidence.json');assert a5['status']=='PASS'
a6=read('evaluation/submission_p1_allocation_structure/paired_summary.json')
read('evaluation/submission_p1_allocation_structure/protocol_lock.json');read('evaluation/gate1_step3_compatibility_retention_contract_audit/summary.json')
read('evaluation/p2_shared_cohort_feasibility_audit/feasibility_summary.json')
save('summary.json',dict(status='INCONCLUSIVE',historical_statistics=counts,qualified_new_native_shared_snapshots=0,qualification_scope='本轮核对的冻结输入、历史快照与运行日志；不声称所有未检查数据都不存在共享',lmcache_token_pairs=16,A5_census=a5['dual_branch']['census'],A6_scenario_only=a6))
csvsave('snapshot_inventory.csv',inventory);csvsave('historical_policy_differences.csv',comparisons);csvsave('historical_exact_opt.csv',exact);save('openhands_barrier_checks.json',correctness)
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
save('evidence_manifest.json',dict(source_sha256=hashes,source_integrity='PASS',new_selector_runs=0,new_gpu_runs=0))
print(json.dumps(counts,ensure_ascii=False,indent=2));print('共同 barrier 核对',len(correctness),all(x['requests_complete'] and x['boundary_no_future'] and x['abort_reason'] is None for x in correctness))
