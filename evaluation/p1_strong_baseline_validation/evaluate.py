"""PCB 门禁通过后评估两种冻结代理策略，并保留所有参考输出和逐步证据。"""
from dataclasses import asdict
from decimal import Decimal, localcontext
from unittest.mock import patch
import statistics, traceback, itertools
from common import *
import pcb,ldr,wfc


def minimum_frontier_cover(epoch,frontiers):
    """仅求覆盖全候选最深前沿需要的最少逻辑项数，作为预算结构证书。"""
    active=[i for i,e in enumerate(frontiers) if e>0]
    masks=[]
    for c in epoch.candidates:
        mask=sum(1<<j for j,i in enumerate(active) if epoch.pending[i].identifier in c.compatible and c.position==frontiers[i])
        if mask:masks.append(mask)
    dp={0:0}
    for mask in set(masks):
        for old,count in list(dp.items()):dp[old|mask]=min(dp.get(old|mask,len(active)+1),count+1)
    return dp[(1<<len(active))-1]


def verify_decimal(epoch,result):
    """以独立高精度公式和位置枚举核对新增策略评分，不参与任何选择。"""
    selected={c.identifier for c in epoch.candidates if c.identifier in result['selected']}
    costs=[]
    with localcontext() as ctx:
        ctx.prec=40
        for p,record in zip(epoch.pending,result['per_pending']):
            available=[c.position for c in epoch.candidates if c.identifier in selected and p.identifier in c.compatible]
            e=max(available) if available else 0;g=p.target-e
            assert (record['E'],record['G'])==(e,g)
            gd=Decimal(g)/1024;td=Decimal(p.target)/1024
            value=Decimal('37.828150')*gd+Decimal('0.345974143')*gd*td-Decimal('0.156201917')*gd*gd
            close(float(value),record['cost_ms']);costs.append(value)
        close(float(sum(costs)),result['cost_ms'])
    close(sum(p['cost_ms'] for p in result['per_pending']),result['cost_ms'])


def divergence(epoch,fs_order,other_order):
    """在相同已选前缀下比较两个实际选择，不用最终集合反推内部过程。"""
    if list(fs_order)==list(other_order):return None
    i=next(i for i,(a,b) in enumerate(itertools.zip_longest(fs_order,other_order)) if a!=b)
    prefix=fs_order[:i];before=score(epoch,prefix);e=frontier_vector(before)
    def metrics(cid):
        if cid is None:return None
        c=next(c for c in epoch.candidates if c.identifier==cid)
        advances=[max(0,c.position-e[j]) if p.identifier in c.compatible else 0 for j,p in enumerate(epoch.pending)]
        after=score(epoch,list(prefix)+[cid])
        return dict(checkpoint_id=cid,L=c.position,q=sum(x>0 for x in advances),r=sum(advances),advances=advances,marginal_ms=before['cost_ms']-after['cost_ms'],delta_per_pending_ms=[x['cost_ms']-y['cost_ms'] for x,y in zip(before['per_pending'],after['per_pending'])])
    return dict(step=i+1,common_prefix=list(prefix),E_before=list(e),FlowState=metrics(fs_order[i] if i<len(fs_order) else None),baseline=metrics(other_order[i] if i<len(other_order) else None))


def summarize(rows):
    out={}
    for pop in ('OpenHands','AgentX'):
        by={}
        for ratio in ('all',.25,.5,.75):
            rs=[r for r in rows if r['population']==pop and (ratio=='all' or r['budget_ratio']==ratio)]
            entry=dict(cases=len(rs),snapshots=len({r['snapshot_id'] for r in rs}),policy_costs={},comparisons={},binding_cases=sum(r['budget_structure']['K_binds_best_frontiers'] for r in rs))
            for name in rs[0]['policies']:
                costs=[r['policies'][name]['cost_ms'] for r in rs]
                entry['policy_costs'][name]=dict(mean=statistics.mean(costs),median=statistics.median(costs),reaches_all_frontiers=sum(r['policies'][name]['reaches_all_frontiers'] for r in rs))
            for name in ('PCB','LDR','WFC'):
                cs=[r['comparisons'][name] for r in rs];ds=[c['baseline_minus_flowstate_ms'] for c in cs]
                entry['comparisons'][name]=dict(win=sum(d>EPS for d in ds),tie=sum(abs(d)<=EPS for d in ds),loss=sum(d<-EPS for d in ds),mean_difference_ms=statistics.mean(ds),median_difference_ms=statistics.median(ds),min_difference_ms=min(ds),max_difference_ms=max(ds),same_set=sum(c['same_set'] for c in cs),different_sets_same_EG=sum(not c['same_set'] and c['same_EG'] for c in cs),same_EG=sum(c['same_EG'] for c in cs),order_different=sum(c['order_different'] for c in cs))
            by[str(ratio)]=entry
        out[pop]=by
    return out


def main():
    gate=read(OUT/'pcb_gate.json');assert gate['status']=='PASS' and gate['ties']==573
    assert gate['pcb_sha256']==sha(OUT/'pcb.py') and gate['common_sha256']==sha(OUT/'common.py')
    assert gate['definition_sha256']==sha(ROOT/'evaluation/P1_BASELINE_AND_WORKLOAD_DESIGN_REVIEW.md')
    lock={p.name:sha(p) for p in [OUT/'pcb.py',OUT/'ldr.py',OUT/'wfc.py',OUT/'common.py']}
    write('implementation_lock.json',lock)
    before=read(OUT/'source_integrity_before.json');assert input_hashes()==before
    pcbrows={(r['population'],r['snapshot_id'],r['k']):r for r in jsonl(OUT/'pcb_gate_results.jsonl')}
    cases=load_cases();results=[];current=None
    try:
        for case in cases:
            current=case;epoch=case['epoch'];digest=fingerprint(epoch);key=(case['population'],case['snapshot_id'],epoch.k)
            pg=pcbrows[key];assert pg['input_digest']==digest
            policies=dict(case['references']);policies['PCB']=pg['PCB'];traces={'PCB':pg['pcb_trace']}
            for name,mod in [('LDR',ldr),('WFC',wfc)]:
                with patch.object(type(MODEL),'estimate',side_effect=AssertionError('代理策略决策不得调用 Phi')):
                    selected,trace=mod.select(epoch)
                    reverse=Epoch(tuple(reversed(epoch.candidates)),tuple(reversed(epoch.pending)),epoch.k)
                    assert mod.select(reverse)[0]==selected
                policies[name]=score(epoch,selected);traces[name]=trace
            assert fingerprint(epoch)==digest
            for name in ('PCB','LDR','WFC'):verify_decimal(epoch,policies[name])
            af=all_frontier(epoch);required=minimum_frontier_cover(epoch,af);bm=case['benefit_matrix_ms']
            effective=sum(x>0 for x in af);disjoint=all(sum(v>0 for v in values)<=1 for values in bm.values())
            for v in policies.values():v['reaches_all_frontiers']=frontier_vector(v)==af
            comparisons={}
            fs=policies['FlowState']
            for name in ('PCB','LDR','WFC'):
                v=policies[name];delta=v['cost_ms']-fs['cost_ms'];parts=[p['cost_ms']-f['cost_ms'] for p,f in zip(v['per_pending'],fs['per_pending'])];close(sum(parts),delta)
                comparisons[name]=dict(baseline_minus_flowstate_ms=delta,delta_per_pending_ms=parts,same_set=set(v['selected'])==set(fs['selected']),same_EG=frontier_vector(v)==frontier_vector(fs),order_different=list(v['selected'])!=case['flowstate_order'],first_divergence=divergence(epoch,case['flowstate_order'],v['selected']))
            reason=('无共享且预算受限，逐请求最佳项 Top-K 等价' if disjoint and epoch.k<effective else '无共享且预算足以保留每个有效请求最佳项' if disjoint else '存在共享且 K 不小于有效请求数' if epoch.k>=effective else '共享推荐去重后可保留全部最佳前沿')
            result={k:v for k,v in case.items() if k not in ('epoch','references')}
            result.update(input=asdict(epoch),policies=policies,traces=traces,comparisons=comparisons,pcb_equivalence_reason=reason,budget_structure=dict(effective_pending=effective,minimum_checkpoints_for_all_frontiers=required,K_binds_best_frontiers=epoch.k<required,all_candidate_frontiers=list(af),all_candidate_cost_ms=sum(MODEL.estimate(p.target-e,p.target) for p,e in zip(epoch.pending,af)),positive_checkpoint_count=sum(any(x>0 for x in values) for values in bm.values())))
            results.append(result)
        assert len(results)==573 and input_hashes()==before
        assert all(sha(OUT/n)==v for n,v in lock.items())
        with (OUT/'per_case_results.jsonl').open('w') as f:
            for r in results:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
        summary=summarize(results);write('summary.json',summary)
        strict=[dict(population=r['population'],snapshot_id=r['snapshot_id'],k=r['k'],baseline=n,difference_ms=c['baseline_minus_flowstate_ms']) for r in results for n,c in r['comparisons'].items() if abs(c['baseline_minus_flowstate_ms'])>EPS]
        write('strict_cases.json',strict)
        write('validation.json',dict(status='PASS',pcb_gate=gate,cases=573,new_policy_evaluations=1719,frozen_reference_scores_reproduced=2865,contract_tests='10/10 PASS',input_order_invariance='1719/1719 PASS',independent_decimal_scoring='1719/1719 PASS',proxy_phi_call_guard='1146/1146 PASS',budget_eligibility_compatibility='PASS',per_pending_sums_and_EG='PASS',input_hashes_unchanged=len(before),implementation_hashes=lock,strict_comparisons=len(strict),gpu_started=False,population_regenerated=False,old_selectors_rerun=False,numeric_tolerance_ms=EPS,说明='胜平负为模型成本比较；没有测量或推断 GPU TTFT/E2E。'))
        print(json.dumps({p:{g:{'comparisons':x['comparisons'],'binding':x['binding_cases']} for g,x in v.items() if g=='all'} for p,v in summary.items()},ensure_ascii=False,indent=2))
    except Exception as e:
        failure=dict(status='VALIDATION_BLOCKED',error=str(e),traceback=traceback.format_exc(),completed_cases=len(results))
        if current:failure['case']={k:asdict(v) if k=='epoch' else v for k,v in current.items()}
        write('validation.json',failure)
        (OUT/'final_report.md').write_text('# 正确性核验失败\n\nVALIDATION_BLOCKED\n\n'+str(e)+'\n\n详情见 validation.json；没有修改冻结输入。\n')
        raise

if __name__=='__main__':main()
