"""只读核对冻结分配结果，输出逐案例机制归因；不调用引擎或重新构造正式总体。"""
from pathlib import Path
import sys, json, hashlib, statistics, itertools
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from flowstate.recovery_model import RecoveryCostModel
from flowstate.optimizer import GlobalOptimizer
OUT = Path(__file__).resolve().parent
R = ROOT / 'evaluation/runtime_artifacts'
OH = R / 'rq3_formal_policy_eval_20260904_110011'
SN = R / 'rq3_openhands_main_formal_20260904_001017'
AX = ROOT / 'rq3_agentx_policy_eval_20260906_214956'
AB = R / 'rq5_frozen_snapshot_ablation_20260908_110726'
AUDIT = R / 'rq3_sanity_structure_audit_20260904_121500'
EPS = 1e-9
MODEL = RecoveryCostModel()

def read(p):
    return json.loads(p.read_text())

def lines(p):
    return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]

def dump(p, x):
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2)+'\n')

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def close(a, b):
    assert abs(a-b) <= EPS, (a,b,a-b)

def stats(xs):
    if not xs:
        return {'n':0}
    xs=sorted(xs)
    def q(p):
        f=(len(xs)-1)*p;i=int(f);return xs[i]+(xs[min(i+1,len(xs)-1)]-xs[i])*(f-i)
    return dict(n=len(xs),mean=statistics.mean(xs),median=statistics.median(xs),min=xs[0],p25=q(.25),p75=q(.75),p95=q(.95),max=xs[-1])

def wtl(xs):
    return {'win':sum(v>EPS for v in xs),'tie':sum(abs(v)<=EPS for v in xs),'loss':sum(v < -EPS for v in xs)}

sources = sorted(set([p for d in (OH,SN,AX,AB,AUDIT) for p in d.rglob('*') if p.is_file()]+[ROOT/p for p in ('flowstate/optimizer.py','flowstate/recovery_model.py','flowstate/state_catalog.py','evaluation/sota_policies.py','evaluation/rq3_formal_policy_evaluation.py','evaluation/rq3_agentx_formal_policy_evaluation.py','evaluation/rq3_sanity_structure_audit.py','evaluation/rq5_frozen_snapshot_ablation.py')]))
before={str(p):digest(p) for p in sources}
snapshots={}
for p in sorted((SN/'snapshots').glob('g*.json')):
    s=json.loads(read(p)['canonical_snapshot']);snapshots[s['snapshot_id']]=(s,str(p))
manifest={x['snapshot_id']:x for x in read(AX/'formal_snapshot_manifest.json')}
ablation={(x['population'],x['snapshot_id'],x['k']):x for x in lines(AB/'per_case_results.jsonl')}
records=[]

for pop, path in [('OpenHands',OH/'per_snapshot_results.jsonl'),('AgentX',AX/'per_case_results.jsonl')]:
    rows=lines(path)
    assert len(rows)==(504 if pop=='OpenHands' else 69)
    assert len({x['snapshot_id'] for x in rows})==(168 if pop=='OpenHands' else 23)
    for line,row in enumerate(rows,1):
        sid=row['snapshot_id'];k=row['k'];md=row['mechanism_diagnostics'];raw=md['standalone_benefits'];ids=sorted(raw)
        pends=row['policies']['FlowState']['per_continuation'];pids=[p['continuation_id'] for p in pends];targets=[p['target_tokens'] for p in pends]
        compat=md['compatibility_matrix'];bs={c:[next(v['benefit_ms'] for v in raw[c]['per_pending'] if v['continuation_id']==p) for p in pids] for c in ids}
        empty=[MODEL.estimate(t,t) for t in targets];empty_total=sum(empty)
        close(empty_total,row['c_empty']);assert len(ids)==row['candidate_count']
        ab=ablation[(pop,sid,k)];assert ab['snapshot_digest']==row['snapshot_digest'];assert ab['candidate_count']==len(ids) and ab['pending_count']==len(pids)
        assert row['snapshot_digest_before']==row['snapshot_digest_after']
        for c in ids:
            close(sum(bs[c]),raw[c]['aggregate_benefit_ms'])
            for j,p in enumerate(pids):
                lc=raw[c]['token_pos'];expected=empty[j]-MODEL.estimate(targets[j]-lc,targets[j]) if compat[c][p] else 0.
                close(expected,bs[c][j])
        def score(selected):
            out=[]
            for j,p in enumerate(pids):
                e=max([raw[c]['token_pos'] for c in selected if compat[c][p]]+[0]);g=targets[j]-e
                out.append(dict(continuation_id=p,T=targets[j],E=e,G=g,cost_ms=MODEL.estimate(g,targets[j])))
            return out
        def cost(selected):
            return sum(p['cost_ms'] for p in score(selected))
        policies={name:dict(selected=v['selected_checkpoint_ids'],cost_ms=v['total_recovery_cost_ms']) for name,v in row['policies'].items()}
        st=ab['variants']['without_set_dependent'];fs=ab['variants']['full']
        policies['StandaloneTopK']=dict(selected=st['selected_checkpoint_ids'],cost_ms=st['common_cost_ms'])
        for name,v in policies.items():
            assert len(v['selected'])<=k and len(v['selected'])==len(set(v['selected'])) and set(v['selected'])<=set(ids)
            v['per_pending']=score(v['selected']);close(cost(v['selected']),v['cost_ms'])
            u=[max([bs[c][j] for c in v['selected']]+[0.]) for j in range(len(pids))]
            close(empty_total-sum(u),v['cost_ms'])
            if name in row['policies']:
                old={p['continuation_id']:p for p in row['policies'][name]['per_continuation']}
                for p in v['per_pending']:
                    assert p['E']==old[p['continuation_id']]['executable_frontier_tokens']
                    assert p['G']==old[p['continuation_id']]['recovery_gap_tokens'];close(p['cost_ms'],old[p['continuation_id']]['recovery_cost_ms'])
        close(fs['common_cost_ms'],policies['FlowState']['cost_ms']);assert set(fs['selected_checkpoint_ids'])==set(policies['FlowState']['selected'])
        order=md['flowstate_greedy_trace']['selection_order'];assert order==fs['selection_order']
        # 复用现有贪心选择器，仅把成本读取替换为冻结兼容矩阵上的同一成本公式。
        class FrozenOptimizer(GlobalOptimizer):
            def _recovery_cost(self, continuations, selected):
                return cost([c.checkpoint_id for c in selected])
        candidates=[SimpleNamespace(checkpoint_id=c,memory_bytes=1,recurrent_resident=True) for c in ids]
        replay=FrozenOptimizer(MODEL).select((),candidates,k)
        assert [c.checkpoint_id for c in replay.selected]==order
        static_scores={c:sum(bs[c]) for c in ids}
        ranking=sorted([c for c in ids if static_scores[c]>EPS],key=lambda c:(-static_scores[c],c))
        assert set(ranking[:k])==set(st['selected_checkpoint_ids'])
        # 消融文件另外记录按总成本差排序的展示顺序，保留其原始浮点求和口径。
        display_scores={c:empty_total-cost([c]) for c in ids}
        display_ranking=sorted([c for c in ids if display_scores[c]>EPS],key=lambda c:(-display_scores[c],c))
        assert display_ranking[:k]==st['selection_order']
        steps=[];prefix=[];u=[0.]*len(pids)
        for i,c in enumerate(order):
            marginals={a:sum(max(bs[a][j]-u[j],0.) for j in range(len(pids))) for a in ids if a not in prefix}
            delta=cost(prefix)-cost(prefix+[c]);close(delta,marginals[c]);close(delta,md['flowstate_greedy_trace']['delta_sequence'][i])
            contributions=[max(bs[c][j]-u[j],0.) for j in range(len(pids))]
            steps.append(dict(step=i+1,prefix=list(prefix),u_before_ms=list(u),all_marginals_ms=marginals,chosen=c,delta_ms=delta,per_pending_gain_ms=contributions))
            prefix.append(c);u=[max(u[j],bs[c][j]) for j in range(len(pids))]
            steps[-1]['u_after_ms']=list(u)
        terminal_u=[max([bs[c][j] for c in order]+[0.]) for j in range(len(pids))]
        terminal_marginals={c:sum(max(bs[c][j]-terminal_u[j],0.) for j in range(len(pids))) for c in ids if c not in order}
        joint_rank_reversals=[]
        for step in steps:
            chosen=step['chosen'];chosen_parts=step['per_pending_gain_ms']
            if sum(v>EPS for v in chosen_parts)<2:continue
            for c,m in step['all_marginals_ms'].items():
                parts=[max(bs[c][j]-step['u_before_ms'][j],0.) for j in range(len(pids))]
                if step['delta_ms']>m+EPS and max(chosen_parts)<max(parts)-EPS:
                    joint_rank_reversals.append(dict(step=step['step'],chosen=chosen,alternative=c,chosen_parts=chosen_parts,alternative_parts=parts))
        diff=set(order)!=set(st['selected_checkpoint_ids']);first=None
        if diff:
            st_order=st['selection_order']
            index=next(i for i,(a,b) in enumerate(itertools.zip_longest(order,st_order)) if a!=b)
            retained=order[:index];u0=[max([bs[c][j] for c in retained]+[0.]) for j in range(len(pids))]
            def info(c):
                if c is None:return None
                gain=[max(bs[c][j]-u0[j],0.) for j in range(len(pids))]
                return dict(checkpoint_id=c,standalone_ms=static_scores[c],marginal_ms=sum(gain),per_pending_gain_ms=gain,positive_pending=[pids[j] for j,v in enumerate(gain) if v>EPS])
            first=dict(step=index+1,retained_set=retained,u_before_ms=u0,flowstate=info(order[index] if index<len(order) else None),standalone=info(st_order[index] if index<len(st_order) else None))
        final_redundant={name:[c for c in v['selected'] if abs(cost([a for a in v['selected'] if a!=c])-v['cost_ms'])<=EPS] for name,v in policies.items()}
        # 下列数值明确相对“最终集合去掉自身”，不是基线内部选择过程。
        selected_shared=[c for c in order if sum(v>EPS for v in bs[c])>1]
        sharing_steps=[s['step'] for s in steps if sum(v>EPS for v in s['per_pending_gain_ms'])>1]
        shared_essential=[c for c in selected_shared if c not in final_redundant['FlowState']]
        structural_shared=[c for c in ids if sum(compat[c].values())>1]
        if pop=='OpenHands':
            sn,snpath=snapshots[sid];assert len(sn['eligible_candidates'])==len(ids)
            realc={c['checkpoint_id']:c for c in sn['eligible_candidates']};realp=sn['pending_continuations']
            assert set(realc)==set(ids)
            for p in realp:assert min(p['anchor_pos'],p['resident_fa_frontier'])==targets[pids.index(p['continuation_id'])]
            for c in ids:
                assert realc[c]['token_pos']==raw[c]['token_pos'] and realc[c]['recurrent_resident']
                for p in realp:
                    cr=realc[c];cp=cr['lineage_path'];pp=p['lineage_path']
                    assert compat[c][p['continuation_id']]==(cr['workflow_id']==p['workflow_id'] and pp[:len(cp)]==cp and cr['token_pos']<=min(p['anchor_pos'],p['resident_fa_frontier']))
            meta={x['checkpoint_id']:x for x in sn['candidate_metadata']};rec={c:meta[c]['last_access_order'] for c in ids};eff={c:meta[c]['marconi_flop_saved']/realc[c]['memory_bytes'] for c in ids}
            def norm(xs):
                lo=min(xs.values());hi=max(xs.values());return {c:(v-lo)/(hi-lo) if hi>lo else 0. for c,v in xs.items()}
            nr=norm(rec);ne=norm(eff);utility={c:nr[c]+sn['marconi_alpha']*ne[c] for c in ids}
            marorder=sorted(ids,key=lambda c:(-utility[c],c))[:k]
            assert set(marorder)==set(policies['Marconi']['selected'])
            metadata=dict(source=snpath,candidates=realc,pending=realp,marconi_alpha=sn['marconi_alpha'],marconi_raw=meta,marconi_utility=utility,marconi_ranking=sorted(ids,key=lambda c:(-utility[c],c)))
            workflows=len({p['workflow_id'] for p in realp})
        else:
            fm=manifest[sid];assert fm['candidate_count']==len(ids)
            for c in ids:assert {p for p in pids if compat[c][p]}==set(fm['objective_usable_pending_ids'][c])
            assert row['common_input_verified'];assert len(set(row['policy_input_digests'].values()))==1
            structural_shared=[c for c,v in fm['formal_compatible_pending_ids'].items() if len(v)>1]
            metadata=dict(source=str(AX/'formal_snapshot_manifest.json'),formal_compatibility=fm,marconi_raw='UNKNOWN：正式逐案例文件未记录原始时序与计算量信号。',actual_lineage_paths='UNKNOWN：正式文件保留兼容关系，但未保留可直接读取的完整原始血缘路径。',encoded_workflow_warning='候选 workflow_id 含兼容性编码，不能当成独立真实工作流。')
            workflows=None
        improvements={name:v['cost_ms']-policies['FlowState']['cost_ms'] for name,v in policies.items() if name!='FlowState'}
        for name,v in policies.items():
            if name!='FlowState':
                v['delta_per_pending_ms']=[p['cost_ms']-f['cost_ms'] for p,f in zip(v['per_pending'],policies['FlowState']['per_pending'])];close(sum(v['delta_per_pending_ms']),improvements[name])
        tie_reason=None
        if abs(improvements['StandaloneTopK'])<=EPS:
            if not diff:tie_reason='选中集合相同'
            elif all(a['E']==b['E'] for a,b in zip(policies['FlowState']['per_pending'],policies['StandaloneTopK']['per_pending'])):
                tie_reason='不同集合但每个待续请求的可执行前沿相同'
            else:tie_reason='不同前沿但总成本抵消'
        tags=dict(A_overlap=any(sum(bs[c][j]>EPS for c in ids)>1 for j in range(len(pids))),A_standalone_redundant=bool(final_redundant['StandaloneTopK']),A_marconi_redundant=bool(final_redundant['Marconi']),B_independent_workflows=(workflows>1 if workflows is not None else None),B_budget_binding_independent=((workflows>1 and k<sum(any(bs[c][j]>EPS for c in ids) for j in range(len(pids)))) if workflows is not None else None),C_structural=bool(structural_shared),C_positive=any(sum(v>EPS for v in bs[c])>1 for c in ids),C_multiple_positive_step=bool(sharing_steps),C_selected_essential=bool(shared_essential),C_joint_rank_reversal=bool(joint_rank_reversals),D_standalone_diff_tie=diff and abs(improvements['StandaloneTopK'])<=EPS,D_marconi_diff_tie=set(policies['Marconi']['selected'])!=set(order) and abs(improvements['Marconi'])<=EPS,E_marconi_tie=abs(improvements['Marconi'])<=EPS,E_marconi_small_positive=EPS<improvements['Marconi']<=1.,E_marconi_adverse=improvements['Marconi'] < -EPS)
        records.append(dict(population=pop,snapshot_id=sid,budget_ratio=row['budget_ratio'],k=k,candidate_count=len(ids),pending_count=len(pids),source_file=str(path),source_line=line,snapshot_digest=row['snapshot_digest'],pending=pends,candidates=raw,compatibility=compat,benefit_matrix_ms=bs,pending_order=pids,metadata=metadata,policies=policies,flowstate_steps=steps,terminal_marginals_ms=terminal_marginals,joint_rank_reversals=joint_rank_reversals,standalone_ranking=ranking,standalone_display_ranking=display_ranking,first_divergence=first,selected_set_different=diff,improvement_ms=improvements,tags=tags,standalone_tie_reason=tie_reason,all_candidates_cost_ms=cost(ids),final_removable_zero=final_redundant,structural_shared=structural_shared,selected_shared=selected_shared,shared_essential=shared_essential,sharing_steps=sharing_steps))

summary={}
for pop in ('OpenHands','AgentX'):
    rs=[r for r in records if r['population']==pop]
    entry=dict(cases=len(rs),snapshots=len({r['snapshot_id'] for r in rs}),comparisons={},by_budget={},classifications={},ties={})
    for b in ('Marconi','LRU','LFU','StandaloneTopK'):
        vals=[r['improvement_ms'][b] for r in rs];entry['comparisons'][b]=dict(wtl=wtl(vals),improvement_ms=stats(vals),different_sets=sum(set(r['policies'][b]['selected'])!=set(r['policies']['FlowState']['selected']) for r in rs),positive_ids=[dict(snapshot_id=r['snapshot_id'],k=r['k']) for r in rs if r['improvement_ms'][b]>EPS],negative_ids=[dict(snapshot_id=r['snapshot_id'],k=r['k']) for r in rs if r['improvement_ms'][b]<-EPS])
    for ratio in (.25,.5,.75):
        rr=[r for r in rs if r['budget_ratio']==ratio];entry['by_budget'][str(ratio)]={b:dict(wtl=wtl([r['improvement_ms'][b] for r in rr]),improvement_ms=stats([r['improvement_ms'][b] for r in rr])) for b in ('Marconi','LRU','LFU','StandaloneTopK')}
    for tag in rs[0]['tags']:
        rr=[r for r in rs if r['tags'][tag]];entry['classifications'][tag]=dict(count=len(rr),unknown_count=sum(r['tags'][tag] is None for r in rs),fraction=len(rr)/len(rs),vs_marconi_ms=stats([r['improvement_ms']['Marconi'] for r in rr]),vs_standalone_ms=stats([r['improvement_ms']['StandaloneTopK'] for r in rr]))
    for reason in sorted({r['standalone_tie_reason'] for r in rs if r['standalone_tie_reason']}):
        rr=[r for r in rs if r['standalone_tie_reason']==reason];entry['ties'][reason]=dict(count=len(rr),reaches_all_candidate_frontier=sum(abs(r['policies']['FlowState']['cost_ms']-r['all_candidates_cost_ms'])<=EPS for r in rr),zero_cost=sum(abs(r['policies']['FlowState']['cost_ms'])<=EPS for r in rr))
    summary[pop]=entry
assert summary['AgentX']['comparisons']['StandaloneTopK']['different_sets']==24
assert summary['AgentX']['comparisons']['StandaloneTopK']['wtl']==dict(win=2,tie=67,loss=0)
assert len(records)==573
for r in records:
    if r['improvement_ms']['StandaloneTopK']>EPS:
        assert r['first_divergence']['standalone']['marginal_ms']<=EPS
        assert len(r['first_divergence']['flowstate']['positive_pending'])==1
    assert r['standalone_ranking']==r['standalone_display_ranking']
frozen_wtl=read(AUDIT/'win_tie_loss.json')
for ratio,entry in frozen_wtl.items():
    for b in ('LRU','LFU','Marconi'):
        assert summary['OpenHands']['by_budget'][ratio][b]['wtl']=={k:entry['FlowState_vs_'+b][k] for k in ('win','tie','loss')}
for pop,root in [('OpenHands',OH),('AgentX',AX)]:
    agg=read(root/'aggregate_results.json')
    for ratio in (.25,.5,.75):
        key='ratio_'+(str(ratio) if pop=='OpenHands' else str(int(ratio*100)))
        for b in ('FlowState','Marconi','LRU','LFU'):
            close(statistics.mean(r['policies'][b]['cost_ms'] for r in records if r['population']==pop and r['budget_ratio']==ratio),agg[key]['C_'+b]['mean'])
with (OUT/'all_cases.jsonl').open('w') as f:
    for r in records:f.write(json.dumps(r,ensure_ascii=False)+'\n')
dump(OUT/'summary.json',summary)
after={str(p):digest(p) for p in sources}
assert before==after
dump(OUT/'integrity.json',dict(status='PASS',说明='所有读取的冻结目录和相关实现文件在分析前后逐文件 SHA-256 相同。未运行引擎或实验，未重建总体。',files=len(sources),before=before,after=after,validation=dict(cases=573,policy_objectives=2865,original_greedy_replay=573,standalone_selected_sets=573,openhands_marconi_replay=504,compatibility_and_targets='PASS',cost_and_benefit_decomposition='PASS',tolerance_ms=EPS)))
print(json.dumps({p:{'cases':s['cases'],'comparisons':{b:{'wtl':v['wtl'],'different_sets':v['different_sets'],'mean':v['improvement_ms']['mean']} for b,v in s['comparisons'].items()},'classes':{k:v['count'] for k,v in s['classifications'].items()},'ties':s['ties']} for p,s in summary.items()},ensure_ascii=False,indent=2))
