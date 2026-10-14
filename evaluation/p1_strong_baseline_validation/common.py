"""读取冻结分配记录并用不变的兼容矩阵和正式 Phi 统一评分。"""
from dataclasses import dataclass, asdict
from pathlib import Path
import sys, json, hashlib
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from flowstate.recovery_model import RecoveryCostModel
OUT=Path(__file__).resolve().parent
P0=ROOT/'evaluation/p0_allocation_mechanism_audit'
R=ROOT/'evaluation/runtime_artifacts'
OH=R/'rq3_formal_policy_eval_20260904_110011'
AX=ROOT/'rq3_agentx_policy_eval_20260906_214956'
AB=R/'rq5_frozen_snapshot_ablation_20260908_110726'
EPS=1e-9
MODEL=RecoveryCostModel()

@dataclass(frozen=True)
class Candidate:
    """只暴露冻结 ID、位置和正式可用兼容集合。"""
    identifier:str
    position:int
    compatible:tuple[str,...]

@dataclass(frozen=True)
class Pending:
    """只暴露当前请求 ID 与冻结 planning target。"""
    identifier:str
    target:int

@dataclass(frozen=True)
class Epoch:
    """选择器无法访问旧策略结果或运行时性能的不可变输入。"""
    candidates:tuple[Candidate,...]
    pending:tuple[Pending,...]
    k:int


def read(p):return json.loads(Path(p).read_text())
def jsonl(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def write(name,value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def close(a,b):
    if abs(a-b)>EPS:raise ValueError(f'成本不一致：{a} 与 {b}，差值 {a-b}')
def fingerprint(epoch):return hashlib.sha256(json.dumps(asdict(epoch),sort_keys=True).encode()).hexdigest()

def input_hashes():
    """覆盖 P0 已审计冻结源、P0 全产物、正式定义和本轮使用的核心语义。"""
    prior=read(P0/'integrity.json');paths=set(map(Path,prior['before']))
    paths.update(p for p in P0.iterdir() if p.is_file())
    paths.update(ROOT/p for p in ['evaluation/P1_BASELINE_AND_WORKLOAD_DESIGN_REVIEW.md','flowstate/executable_state.py','flowstate/workflow.py'])
    hashes={str(p):sha(p) for p in sorted(paths)}
    for p,v in prior['before'].items():
        if hashes[p]!=v:raise ValueError(f'冻结输入摘要与 P0 不一致：{p}')
    return hashes

def score(epoch,selected):
    """按正式 E=max L、G=T-E、C=sum Phi 的同一语义评分冻结矩阵。"""
    by_id={c.identifier:c for c in epoch.candidates}
    if len(set(selected))!=len(selected) or len(selected)>epoch.k or not set(selected)<=set(by_id):
        raise ValueError('选择集合存在重复、越界或未知候选')
    rows=[]
    for p in epoch.pending:
        e=max([by_id[c].position for c in selected if p.identifier in by_id[c].compatible]+[0])
        g=p.target-e
        rows.append(dict(continuation_id=p.identifier,T=p.target,E=e,G=g,cost_ms=MODEL.estimate(g,p.target)))
    return dict(selected=list(selected),cost_ms=sum(p['cost_ms'] for p in rows),per_pending=rows)

def frontier_vector(result):return tuple(p['E'] for p in result['per_pending'])

def all_frontier(epoch):
    return tuple(max([c.position for c in epoch.candidates if p.identifier in c.compatible]+[0]) for p in epoch.pending)

def benefit_matrix(epoch):
    return {c.identifier:[MODEL.estimate(p.target,p.target)-MODEL.estimate(p.target-c.position,p.target) if p.identifier in c.compatible else 0. for p in epoch.pending] for c in epoch.candidates}

def load_cases():
    """只反序列化既有正式记录，不调用 builder、collector 或旧策略。"""
    p0={(r['population'],r['snapshot_id'],r['k']):r for r in jsonl(P0/'all_cases.jsonl')}
    ab={(r['population'],r['snapshot_id'],r['k']):(i,r) for i,r in enumerate(jsonl(AB/'per_case_results.jsonl'),1)}
    manifest={r['snapshot_id']:r for r in read(AX/'formal_snapshot_manifest.json')}
    cases=[]
    for pop,path in [('OpenHands',OH/'per_snapshot_results.jsonl'),('AgentX',AX/'per_case_results.jsonl')]:
        rows=jsonl(path)
        assert len(rows)==(504 if pop=='OpenHands' else 69)
        assert len({r['snapshot_id'] for r in rows})==(168 if pop=='OpenHands' else 23)
        for lineno,r in enumerate(rows,1):
            key=(pop,r['snapshot_id'],r['k']);old=p0[key];abline,abl=ab[key]
            assert r['snapshot_digest']==old['snapshot_digest']==abl['snapshot_digest']
            assert r['snapshot_digest_before']==r['snapshot_digest_after']
            md=r['mechanism_diagnostics'];raw=md['standalone_benefits'];cm=md['compatibility_matrix']
            pp=r['policies']['FlowState']['per_continuation']
            pending=tuple(Pending(p['continuation_id'],p['target_tokens']) for p in pp)
            candidates=tuple(Candidate(cid,v['token_pos'],tuple(sorted(p for p,yes in cm[cid].items() if yes))) for cid,v in sorted(raw.items()))
            epoch=Epoch(candidates,pending,r['k']);ids={p.identifier for p in pending}
            assert len(candidates)==r['candidate_count'] and len(pending)==r['pending_count']
            assert len(ids)==len(pending) and len({c.identifier for c in candidates})==len(candidates)
            assert r['k']==max(1,int(r['budget_ratio']*len(candidates)))
            for c in candidates:
                assert set(c.compatible)<=ids and all(c.position<=p.target for p in pending if p.identifier in c.compatible)
                assert cm[c.identifier]==old['compatibility'][c.identifier]
            if pop=='AgentX':
                fm=manifest[r['snapshot_id']]
                assert r['common_input_verified'] and len(set(r['policy_input_digests'].values()))==1
                assert set(fm['objective_usable_pending_ids'])==set(raw)
                for c in candidates:assert set(c.compatible)==set(fm['objective_usable_pending_ids'][c.identifier])
            else:
                sn=json.loads(read(old['metadata']['source'])['canonical_snapshot'])
                sc={c['checkpoint_id']:c for c in sn['eligible_candidates']}
                assert set(sc)==set(raw) and len({c['memory_bytes'] for c in sc.values()})==1
                assert all(c['recurrent_resident'] for c in sc.values())
                for c in candidates:
                    cc=sc[c.identifier];assert cc['token_pos']==c.position
                    for p in sn['pending_continuations']:
                        t=min(p['anchor_pos'],p['resident_fa_frontier']);assert t==next(q.target for q in pending if q.identifier==p['continuation_id'])
                        cp=cc['lineage_path'];yes=cc['workflow_id']==p['workflow_id'] and p['lineage_path'][:len(cp)]==cp and c.position<=t
                        assert yes==(p['continuation_id'] in c.compatible)
            bm=benefit_matrix(epoch)
            for cid,values in bm.items():
                saved={x['continuation_id']:x['benefit_ms'] for x in raw[cid]['per_pending']}
                for p,v in zip(pending,values):close(v,saved[p.identifier])
            references={}
            for name,v in r['policies'].items():
                result=score(epoch,v['selected_checkpoint_ids']);close(result['cost_ms'],v['total_recovery_cost_ms'])
                recorded={p['continuation_id']:p for p in v['per_continuation']}
                for p in result['per_pending']:
                    z=recorded[p['continuation_id']]
                    assert (p['T'],p['E'],p['G'])==(z['target_tokens'],z['executable_frontier_tokens'],z['recovery_gap_tokens'])
                    close(p['cost_ms'],z['recovery_cost_ms'])
                references[name]=result
            st=abl['variants']['without_set_dependent'];references['StandaloneTopK']=score(epoch,st['selected_checkpoint_ids']);close(references['StandaloneTopK']['cost_ms'],st['common_cost_ms'])
            close(references['FlowState']['cost_ms'],abl['variants']['full']['common_cost_ms'])
            assert set(references['FlowState']['selected'])==set(abl['variants']['full']['selected_checkpoint_ids'])
            cases.append(dict(population=pop,snapshot_id=r['snapshot_id'],budget_ratio=r['budget_ratio'],k=r['k'],source_file=str(path),source_line=lineno,ablation_source=str(AB/'per_case_results.jsonl'),ablation_line=abline,snapshot_digest=r['snapshot_digest'],epoch=epoch,input_digest=fingerprint(epoch),benefit_matrix_ms=bm,references=references,flowstate_order=md['flowstate_greedy_trace']['selection_order']))
    assert len(cases)==573
    return cases
