"""只读比较两套已冻结兼容定义，不生成新矩阵、不重新评分。"""
import json,csv,hashlib
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
B2=Path('/home/wjg/data/agentx/audits/agentx_runtime_formal_20260906_023649');sources={}
def read(p):
    p=Path(p);b=p.read_bytes();sources[str(p)]={'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)};return b.decode()
def js(p):return json.loads(read(p))
def jl(p):return [json.loads(s) for s in read(p).splitlines()]
def save(n,v):(OUT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
raw=jl(B2/'runtime_snapshots.jsonl');formal=js(ROOT/'rq3_agentx_policy_eval_20260906_214956/formal_snapshot_manifest.json');fm={r['trace_id']:r for r in formal}
prior=js(ROOT/'agentx_runtime_final_audit_20260906_194000/compatibility_exact_set_audit.json');reasons={r['candidate_id']:r for r in prior['mismatched_candidates']}
corrected=jl(ROOT/'agentx_runtime_final_audit_20260906_194000/corrected_runtime_snapshots.jsonl');corr={r['trace_id']:r for r in corrected}
for rel in ['evaluation/rq3_agentx_formal_policy_evaluation.py','evaluation/agentx_runtime_snapshot_collection.py','evaluation/agentx_chain_detection.py','evaluation/agentx_runtime_final_correctness_audit.py','flowstate/state_catalog.py','flowstate/recovery_model.py','flowstate/optimizer.py','flowstate/controller.py','flowstate/adapters/sglang.py','evaluation/control_path_optimized_e2e.py','evaluation/control_path_optimized_client.py','evaluation/rq6_batched_control.py','agentx_runtime_final_audit_20260906_194000/cleanup_semantics_audit.json','evaluation/p3_physical_retention_budget_contract/physical_memory_accounting.md','evaluation/p3_physical_retention_budget_contract/retention_budget_contract.md','evaluation/gate1_step2_native_shared_workflow_feasibility/evidence_manifest.json','evaluation/p4_evaluation_restructuring/runtime_evaluation_plan.md']:read(ROOT/rel)
edges=[];counts=Counter();scenes=[];candidate_counts=Counter();positive_added=set();pool=[]
for rn,r in enumerate(raw,1):
    f=fm[r['trace_id']];assert f['timestamp']==r['t'];cp={c['checkpoint_id']:c for c in r['checkpoints']};pending={p['continuation_id']:p for p in r['pendings']};logical=f['formal_compatible_pending_ids'];usable=f['objective_usable_pending_ids'];assert set(cp)==set(logical)==set(usable)
    assert corr[r['trace_id']]['t']==r['t'];pp=Path(B2/'per_snapshot')/f"snap_{r['trace_id']}_{r['t']:.6f}"/'snapshot.json';detail=js(pp);assert detail['t']==r['t']
    scene=Counter();resolved=Counter()
    for cid,c in cp.items():
        a=set(c['compatible_pending_ids']);b=set(usable[cid]);l=set(logical[cid]);assert (a|b|l)<=set(pending)
        assert b=={pid for pid in l if c['recurrent_prefix_length']<=pending[pid]['planning_target']}
        candidate_counts['same' if a==b else 'different']+=1
        candidate_counts['formal_add_candidates']+=bool(b-a);candidate_counts['raw_only_candidates']+=bool(a-b)
        for pid,p in pending.items():
            av=pid in a;bv=pid in b;lv=pid in l;depth=c['recurrent_prefix_length'];over=depth>p['planning_target'];reason='一致';diff=av!=bv
            if bv and not av:
                reason='逻辑祖先编码新增；本次回放token前缀谓词为假';counts['formal_added_edges']+=1;scene['formal_added_edges']+=1
                if depth>0:positive_added.add((r['trace_id'],cid,pid))
            elif av and not bv:
                counts['raw_only_edges']+=1;scene['raw_only_edges']+=1
                if lv and over:reason='B1逻辑边存在，但物理L超过pending的T'
                else:reason='B1排除：'+reasons.get(cid,{}).get('extra_root_causes',{}).get(pid,'UNKNOWN')
            if diff:resolved[reason]+=1;counts['different_edges']+=1
            if lv and over:counts['b1_edges_removed_by_L_T']+=1
            if diff and over:counts['different_edges_with_L_T_violation']+=1
            if av and bv:counts['common_edges']+=1
            scene['candidate_different']+=0
            edges.append(dict(snapshot_id=f['snapshot_id'],trace_id=r['trace_id'],checkpoint_id=cid,pending_id=pid,mapping_status='EXACT_ID',raw_compatible=av,formal_logical_compatible=lv,formal_compatible=bv,different=diff,recurrent_prefix_length=depth,target=p['planning_target'],L_gt_T=over,reason=reason,raw_prefix_match=av,token_arrays_available='UNKNOWN：原始token数组未随snapshot归档',time_boundary='同一B2快照与观测量；Formal离线重编码',runtime_execution_verified='UNKNOWN',evidence_path=f'{B2}/runtime_snapshots.jsonl:{rn};{ROOT}/rq3_agentx_policy_eval_20260906_214956/formal_snapshot_manifest.json;{pp}',code_evidence='agentx_runtime_snapshot_collection.py:884;rq3_agentx_formal_policy_evaluation.py:373;agentx_runtime_final_correctness_audit.py:87'))
    scene['candidate_different']=sum(set(c['compatible_pending_ids'])!=set(usable[c['checkpoint_id']]) for c in cp.values())
    scenes.append(dict(snapshot_id=f['snapshot_id'],trace_id=r['trace_id'],candidates=len(cp),pending=len(pending),counts=dict(scene),reasons=dict(resolved)))
    census=detail['census_rows'];last=census[-1]['accounting'];pool.append(dict(trace_id=r['trace_id'],candidates=len(cp),last_event=census[-1]['event'],last_accounting=last,min_free=min(x['accounting']['mamba_available'] for x in census),source=str(pp)))
with (OUT/'compatibility_diff.csv').open('w') as h:
    w=csv.DictWriter(h,fieldnames=list(edges[0]));w.writeheader();w.writerows(edges)
raw_results=jl(ROOT/'rq3_agentx_policy_eval_20260906_214956/per_case_results.jsonl');affected=[]
for r in raw_results:
    added={(c,p) for t,c,p in positive_added if t==r['trace_id']};ps={}
    for name,v in r['policies'].items():
        selected=set(v['selected_checkpoint_ids']);hits=[(c,p) for c,p in added if c in selected]
        if hits:ps[name]=hits
    if ps:affected.append(dict(snapshot_id=r['snapshot_id'],budget_ratio=r['budget_ratio'],selected_formal_only_edges=ps))
save('summary.json',dict(candidates=dict(candidate_counts),edges=dict(counts),snapshots=scenes,unmapped=0,unexplained_definition_edges=sum(x['different'] and 'UNKNOWN' in x['reason'] for x in edges),positive_formal_only_edges=len(positive_added),snapshots_positive_formal_only=len({t for t,c,p in positive_added}),affected_conditions=len(affected)))
save('affected_existing_selections.json',affected);save('pool_observations.json',pool)
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==v['sha256'] for p,v in sources.items())
save('evidence_manifest.json',dict(sources=sources,source_unchanged=True,script=str(Path(__file__).resolve()),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),versions='SGLang 0.5.17为历史记录；P3的16个/tmp源码当前缺失；未代用其他版本',status='BLOCKED',no_rescoring=True))
print(json.dumps(dict(candidates=dict(candidate_counts),edges=dict(counts),unexplained=sum(x['different'] and 'UNKNOWN' in x['reason'] for x in edges),positive_formal_only=len(positive_added),affected_conditions=len(affected)),ensure_ascii=False));print(Counter(x['reason'] for x in edges if x['different']))
