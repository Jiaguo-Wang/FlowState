"""只读审查原始 schema 和冻结结构；不重建总体，不调用任何策略或引擎。"""
from pathlib import Path
import hashlib,json,collections,math
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
RAW=Path('/home/wjg/data/agentx/cc-traces-weka-062126/traces.jsonl')
B1=Path('/home/wjg/data/agentx/audits/agentx_runtime_population_census_20260905_010614')
B2=Path('/home/wjg/data/agentx/audits/agentx_runtime_formal_20260906_023649')
used={}
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):
    p=Path(p);data=p.read_bytes();used[str(p)]=hashlib.sha256(data).hexdigest();return json.loads(data)
def textread(p):
    p=Path(p);b=p.read_bytes();used[str(p)]=hashlib.sha256(b).hexdigest();return b.decode()
def write(name,x):(OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
# 所有必读资料和实现仅按文本读取，不导入或执行其函数。
for rel in ['evaluation/P1_BASELINE_AND_WORKLOAD_DESIGN_REVIEW.md','evaluation/p1_strong_baseline_validation/final_report.md','evaluation/p0_allocation_mechanism_audit/final_report.md','evaluation/p0_allocation_mechanism_audit/case_studies.md','evaluation/agentx_structure_audit.py','evaluation/agentx_chain_detection.py','evaluation/agentx_runtime_population_census.py','evaluation/agentx_runtime_snapshot_collection.py','evaluation/agentx_qwen_replay_preflight.py','evaluation/agentx_runtime_final_correctness_audit.py','evaluation/rq3_agentx_formal_policy_evaluation.py','flowstate/workflow.py','flowstate/state_catalog.py','flowstate/executable_state.py','flowstate/recovery_model.py']:
    p=ROOT/rel
    if p.exists():textread(p)
rawhash=hashlib.sha256();traces=[];keys=collections.Counter();types=collections.Counter();topkeys=collections.Counter();invkeys=collections.Counter();examples={};invs=0;missingduration=0;wrappercount=0
with RAW.open('rb') as f:
    for line_number,line in enumerate(f,1):
        rawhash.update(line);r=json.loads(line);topkeys.update(r.keys());events=[];traceinv=0;stack=[(r,'$')]
        while stack:
            node,path=stack.pop();keys.update(node.keys())
            if 'type' in node:
                types[str(node['type'])]+=1
                typ=str(node['type'])
                if typ not in examples:examples[typ]=dict(file=str(RAW),line=line_number,path=path,fields={k:(f'数组长度 {len(v)}' if isinstance(v,list) else '嵌套对象' if isinstance(v,dict) else v) for k,v in node.items()})
            if 'model' in node and 't' in node and ('in' in node or 'out' in node):
                invs+=1;traceinv+=1;invkeys.update(node.keys());t=node.get('t');duration=node.get('api_time')
                if isinstance(t,(int,float)) and isinstance(duration,(int,float)) and math.isfinite(duration) and duration>0:
                    events.extend([(t,1),(t+duration,-1)])
                else:missingduration+=1
            if 'requests' in node and path!='$':wrappercount+=1
            for k,v in node.items():
                if isinstance(v,dict):stack.append((v,path+'.'+k))
                elif isinstance(v,list) and v and isinstance(v[0],dict):
                    stack.extend((x,f'{path}.{k}[{j}]') for j,x in reversed(list(enumerate(v))) if isinstance(x,dict))
        active=maximum=0
        for t,change in sorted(events):active+=change;maximum=max(maximum,active)
        traces.append(dict(trace_id=r['id'],line=line_number,invocations=traceinv,max_api_interval_overlap=maximum))
used[str(RAW)]=rawhash.hexdigest()
assert used[str(RAW)]=='29b6a19e751ff5230771519aab755f80a0f43a4ba9cf96b72d3a6a437ec99276'
raw_summary=dict(trace_count=len(traces),unique_trace_ids=len({r['trace_id'] for r in traces}),invocation_records=invs,invocation_field_counts=dict(invkeys),all_field_counts=dict(keys),top_field_counts=dict(topkeys),request_type_counts=dict(types),nested_request_containers=wrappercount,missing_or_nonpositive_duration=missingduration,api_interval_overlap_trace_count=sum(r['max_api_interval_overlap']>=2 for r in traces),maximum_api_interval_overlap=max(r['max_api_interval_overlap'] for r in traces),说明='API 区间重叠不等于同时待续；仅使用记录中的相对 t 和 api_time，不证明队列或 pending。',examples=examples)
write('raw_schema_audit.json',raw_summary);write('raw_trace_inventory.json',traces)
census=read(B1/'SHARED_CONTEXT_CENSUS.json');protocol=read(B1/'POPULATION_PROTOCOL.json');population=read(B1/'FORMAL_CANDIDATE_POPULATION.json');inputs=read(B1/'INPUTS.json');replay=read(B1/'REPLAY_SEMANTICS.json')
for name in ['manifest.json','runtime_correctness.json','eligibility.json']:read(B2/name)
runtime=[json.loads(s) for s in textread(B2/'runtime_snapshots.jsonl').splitlines()];runtime_by={r['trace_id']:r for r in runtime}
AX=ROOT/'rq3_agentx_policy_eval_20260906_214956';manifest=read(AX/'formal_snapshot_manifest.json');mf={r['snapshot_id']:r for r in manifest}
formal=[json.loads(s) for s in textread(AX/'per_case_results.jsonl').splitlines()]
metrics=[]
for line,r in enumerate(formal,1):
    md=r['mechanism_diagnostics'];candidates=md['standalone_benefits'];pends=r['policies']['FlowState']['per_continuation'];pids=[p['continuation_id'] for p in pends];T={p['continuation_id']:p['target_tokens'] for p in pends};cm=md['compatibility_matrix']
    m=mf[r['snapshot_id']];assert set(candidates)==set(m['objective_usable_pending_ids'])
    covers={c:{p for p in pids if cm[c][p] and candidates[c]['token_pos']>0} for c in candidates}
    for c in candidates:
        assert {p for p in pids if cm[c][p]}==set(m['objective_usable_pending_ids'][c])
        assert all(candidates[c]['token_pos']<=T[p] for p in covers[c])
    best={p:max([candidates[c]['token_pos'] for c in candidates if p in covers[c]]+[0]) for p in pids}
    active=[p for p in pids if best[p]>0];dp={0:0};witness={0:[]}
    for c in sorted(candidates):
        mask=sum(1<<j for j,p in enumerate(active) if p in covers[c] and candidates[c]['token_pos']==best[p])
        for old,count in list(dp.items()):
            new=old|mask
            if count+1<dp.get(new,len(active)+1):dp[new]=count+1;witness[new]=witness[old]+[c]
    target=(1<<len(active))-1;minimum=dp[target];shared=[c for c,v in covers.items() if len(v)>1];private=[c for c,v in covers.items() if len(v)==1]
    trade=[]
    for c in shared:
        eligible={p:[d for d in private if p in covers[d] and candidates[d]['token_pos']>candidates[c]['token_pos']] for p in covers[c]}
        if sum(bool(v) for v in eligible.values())>=2:trade.append(dict(shared=c,deeper_single_pending_candidates=eligible))
    rt=runtime_by[r['trace_id']];rc={c['checkpoint_id']:c for c in rt['checkpoints']};rp={p['continuation_id']:p for p in rt['pendings']}
    assert set(rc)==set(candidates)
    for c in candidates:assert rc[c]['recurrent_prefix_length']==candidates[c]['token_pos'] and rc[c]['recurrent_resident']
    for p in pids:assert T[p]==min(rp[p]['input_token_count'],rp[p]['resident_fa_frontier'])
    metrics.append(dict(snapshot_id=r['snapshot_id'],trace_id=r['trace_id'],source_line=line,budget_ratio=r['budget_ratio'],total_candidates=len(candidates),positive_benefit_candidates=sum(bool(v) for v in covers.values()),effective_pending=len(active),pending_count=len(pids),shared_benefit_candidates=len(shared),branch_specific_candidates=len(private),branch_specific_scope='仅表示正式兼容矩阵中只服务一个 pending；不单独证明真实 lineage',K=r['k'],K_min_frontier=minimum,K_over_K_min_frontier=r['k']/minimum if minimum else None,ratio_note='零分母不适用' if minimum==0 else '',capacity_constrained=r['k']<minimum,all_candidate_frontiers=best,minimum_cover_witness=witness[target],shared_two_deeper_private_patterns=trade,strict_real_cohort_eligibility='UNKNOWN：缺少原始 fork/pending 事件证明'))
write('capacity_certificates.json',metrics)
counts=collections.Counter()
for tr in census['trace_level']:counts.update(tr['counts'])
summary=dict(raw=raw_summary,census=dict(trace_count=len(census['trace_level']),historical_definition_counts=dict(counts),traces_with_eligible_epochs=sum(x['counts'].get('SHARED_CONTEXT_ELIGIBLE',0)>0 for x in census['trace_level']),formal_snapshots=len(population),additional_derived_epochs=sum(x['counts'].get('SHARED_CONTEXT_ELIGIBLE',0) for x in census['trace_level'])-len(population),解释='旧逻辑/上下文门禁计数，不是本轮真实 fork、同时 pending 和 residency 门禁通过数。'),capacity=dict(cases=len(metrics),snapshots=len(manifest),minimum_frontier_histogram=dict(collections.Counter(x['K_min_frontier'] for x in metrics)),constrained_cases=sum(x['capacity_constrained'] for x in metrics),cases_with_two_deeper_private_pattern=sum(bool(x['shared_two_deeper_private_patterns']) for x in metrics)),new_epochs=dict(real_fork_verified='UNKNOWN',simultaneously_pending_verified='UNKNOWN',attention_residency='UNKNOWN',recurrent_residency='UNKNOWN',K_min_frontier='UNKNOWN',capacity_constrained_count='UNKNOWN'),gpu_started=False,baseline_executed=False,new_cohort_created=False)
write('feasibility_summary.json',summary)
write('read_hashes_before.json',used)
print(json.dumps(dict(raw={k:v for k,v in raw_summary.items() if k not in ['examples','all_field_counts']},census=summary['census'],capacity=summary['capacity']),ensure_ascii=False,indent=2))
