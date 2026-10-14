"""只读联接冻结矩阵、历史驻留与原生复制记录，严格区分逻辑与物理证据。"""
import csv,json,hashlib,itertools,sys
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
S1=ROOT/'evaluation/gate1_step1_allocation_equivalence'
B2=Path('/home/wjg/data/agentx/audits/agentx_runtime_formal_20260906_023649')
F1=Path('/home/wjg/code/newfind/artifacts/exp_f1_hybrid_fork_20260814T160725Z')
WP=ROOT/'motivation/artifacts/wp3a_direct_eviction_20260819'
sys.path.insert(0,str(ROOT))
from flowstate.recovery_model import RecoveryCostModel
model=RecoveryCostModel()
manifest={}
def read(p):
    p=Path(p);data=p.read_bytes();manifest[str(p)]={'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)};return data.decode()
def js(p):return json.loads(read(p))
def jl(p):return [json.loads(s) for s in read(p).splitlines()]
def save(n,obj):(OUT/n).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
read(ROOT/'flowstate/recovery_model.py')
rows=list(csv.DictReader(read(S1/'gate1_step1_instance_statistics.csv').splitlines()));mat=jl(S1/'matrices.jsonl')
raw=jl(B2/'runtime_snapshots.jsonl');runtime={r['trace_id']:(i,r) for i,r in enumerate(raw,1)};correct=js(B2/'runtime_correctness.json')
formal=jl(ROOT/'evaluation/p1_strong_baseline_validation/per_case_results.jsonl')
for folder in ['p2_shared_cohort_feasibility_audit','p3_native_instrumentation_feasibility','p3_physical_retention_budget_contract']:
    for p in (ROOT/'evaluation'/folder).glob('*.md'):read(p)
for p in [ROOT/'evaluation/p2_shared_cohort_feasibility_audit/raw_schema_audit.json',ROOT/'evaluation/p2_shared_cohort_feasibility_audit/raw_trace_inventory.json',ROOT/'evaluation/p3_native_instrumentation_feasibility/source_verification.json',ROOT/'evaluation/p3_physical_retention_budget_contract/source_integrity.json',ROOT/'evaluation/agentx_chain_detection.py',ROOT/'evaluation/agentx_runtime_snapshot_collection.py',ROOT/'evaluation/rq3_agentx_formal_policy_evaluation.py',F1/'metadata.json',F1/'findings.md',F1/'verdict.json',F1.parent.parent/'exp-f1/harness/exp_f1.py',F1.parent.parent/'exp-f1/harness/f1_probe.py',WP/'driver.py',WP/'metadata.json',WP/'findings.md',Path('/home/wjg/data/agentx/audits/agentx_runtime_population_census_20260905_010614/REPLAY_SEMANTICS.json')]:read(p)
output=[];cert=[]
for r,m,f in zip(rows,mat,formal):
    assert r['instance_id']==m['instance_id'];ax=r['workload']=='AgentX';n=int(r['num_continuations']);vectors=list(zip(*m['benefit_ms']));k=int(r['budget']);frontiers=[max([l for l,a in zip(m['positions'],row) if a]+[0]) for row in m['compatibility']]
    for pi,t in enumerate(m['targets']):
        for ci,l in enumerate(m['positions']):
            value=model.estimate(t,t)-model.estimate(t-l,t) if m['compatibility'][pi][ci] else 0.
            assert abs(value-m['benefit_ms'][pi][ci])<=1e-9
    active=[i for i,e in enumerate(frontiers) if e>0];dp={0:0}
    for ci,l in enumerate(m['positions']):
        mask=sum(1<<j for j,i in enumerate(active) if m['compatibility'][i][ci] and l==frontiers[i])
        for old,c in list(dp.items()):dp[old|mask]=min(dp.get(old|mask,10**9),c+1)
    km=dp[(1<<len(active))-1];assert km==int(r['k_min_frontier'])
    tradeoff=any(any(x>y+1e-9 for x,y in zip(a,b)) and any(y>x+1e-9 for x,y in zip(a,b)) for a,b in itertools.combinations(vectors,2))
    mismatch='不适用';native='UNKNOWN';sharing='UNKNOWN';rec='RECORDED：正式历史驻留快照';paths=[f"{r['source_file']}:{r['source_line']}",f"{S1/'matrices.jsonl'}:{len(output)+1}"]
    if ax:
        lineno,z=runtime[r['trace_ids']];cp={c['checkpoint_id']:c for c in z['checkpoints']};pp={p['continuation_id']:p for p in z['pendings']};assert set(cp)==set(m['candidate_ids'])
        for ci,cid in enumerate(m['candidate_ids']):assert cp[cid]['recurrent_resident'] and cp[cid]['fa_resident'] and cp[cid]['recurrent_prefix_length']==m['positions'][ci]
        for pi,pid in enumerate(m['pending_ids']):assert min(pp[pid]['input_token_count'],pp[pid]['resident_fa_frontier'])==m['targets'][pi]
        mismatch=sum(set(cp[cid]['compatible_pending_ids'])!={m['pending_ids'][pi] for pi in range(n) if m['compatibility'][pi][ci]} for ci,cid in enumerate(m['candidate_ids']))
        rec='RECORDED：串行BLOCK_SHAPE回放，非原始Agent执行';paths.append(f'{B2}/runtime_snapshots.jsonl:{lineno}')
    shared=int(r['num_shared_checkpoints'])>0;overlap=int(r['num_overlapping_candidate_pairs'])>0;pressure=k<km
    output.append(dict(scenario_id=r['instance_id'],scope='正式冻结条件',source_trace_ids=r['trace_ids'],num_pending=n,num_candidates=int(r['num_candidates']),native_fork_evidence=native,recurrent_state_evidence=rec,physical_sharing_evidence=sharing,shared_positive_benefit=shared,benefit_overlap=overlap,k_min=km,budget_k=k,budget_source='总候选比例逻辑预算；非物理池上限',effective_budget_pressure=pressure,physical_budget_pressure='UNKNOWN',alternative_tradeoff_vectors=tradeoff,nontrivial_allocation=tradeoff,evidence_class='S1' if shared else 'S0',native_ABCD_verified=False,pool_slots=192 if ax else 28,raw_formal_compatibility_different_candidates=mismatch,evidence_paths=';'.join(paths)))
    cert.append(dict(scenario_id=r['instance_id'],frontiers=frontiers,k_min=km,logical_tight_positive_k=list(range(1,km)),说明='只计算逻辑阈值，不选取或执行这些预算；零预算不纳入非平凡保留分析。'))
fork_details=[]
for p in sorted((F1/'raw').glob('fork_*.json')):
    r=js(p);q=r['phases'];g=q['geometry'];children=r['K'];pool=q['fork_census_after']['mamba_pool'];div=q['divergence'];fail=sum(not s['token_equal'] for s in q['correctness']['samples'])
    parentslot=q['parent_state']['leaf_node']['mamba_value'][0];copies=[x for x in q['fork_cow']['copy_records'] if parentslot in x['src']]
    output.append(dict(scenario_id=p.stem,scope='历史受控Runtime Fork',source_trace_ids='UNKNOWN：受控token输入',num_pending='UNKNOWN：记录为已执行子请求',num_candidates='UNKNOWN：无统一待续边界',native_fork_evidence='受控同父输入并发请求；原生Agent fork UNKNOWN',recurrent_state_evidence='RECORDED：父状态、分支槽及COW',physical_sharing_evidence='RECORDED：共享只读源，独立写入副本' if children>1 else '单子请求，不证明多分支共享',shared_positive_benefit='UNKNOWN：缺当前待续P/T',benefit_overlap='UNKNOWN',k_min='UNKNOWN',budget_k='UNKNOWN',budget_source='实测总池471槽；文件K为子请求数',effective_budget_pressure='UNKNOWN',physical_budget_pressure='未观察到；记录有空闲槽且无准入失败',alternative_tradeoff_vectors='UNKNOWN',nontrivial_allocation='UNKNOWN',evidence_class='S2' if children>1 else 'S0',native_ABCD_verified=False,pool_slots=g['mamba_total_slots'],raw_formal_compatibility_different_candidates='不适用',evidence_paths=str(p)))
    fork_details.append(dict(file=str(p),children=children,free_after=pool['available'],total=pool['total'],parent_source_slot=parentslot,parent_copy_records=copies,parent_unchanged=div.get('parent_slot_unchanged','UNKNOWN'),correctness_mismatches=fail,reported_bytes_per_slot=g['mamba_total_bytes_per_slot'],temporal_shape=g['mamba_temporal_shape'],conv_shape=g['mamba_conv_shape_layer0'],distinct=div.get('child_slot_states_distinct','UNKNOWN')))
wp=list(csv.DictReader(read(WP/'raw_runs.csv').splitlines()));read(WP/'driver_events.jsonl')
for i,r in enumerate(wp,2):
    output.append(dict(scenario_id='WP3A:'+r['experiment_id']+':'+r['repetition']+':'+r['case_id'],scope='历史受控组件干预',source_trace_ids=r['measured_rid'],num_pending=1,num_candidates='UNKNOWN：无分配候选集',native_fork_evidence='受控父子/Join路径，非原生Agent待续边界',recurrent_state_evidence='RECORDED：目标驻留与驱逐',physical_sharing_evidence='UNKNOWN：本行仅一个被测consumer',shared_positive_benefit=False,benefit_overlap='UNKNOWN',k_min='UNKNOWN',budget_k='UNKNOWN',budget_source='总池16槽；对称删除一槽而非K预算竞争',effective_budget_pressure='UNKNOWN',physical_budget_pressure='UNKNOWN',alternative_tradeoff_vectors='UNKNOWN',nontrivial_allocation=False,evidence_class='S0',native_ABCD_verified=False,pool_slots=16,raw_formal_compatibility_different_candidates='不适用',evidence_paths=f'{WP}/raw_runs.csv:{i}'))
assert len(output)==600
with (OUT/'scenario_statistics.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(output[0]));w.writeheader();w.writerows(output)
summary={}
for scope in sorted({r['scope'] for r in output}):
    a=[r for r in output if r['scope']==scope];summary[scope]=dict(rows=len(a),classes=dict(Counter(r['evidence_class'] for r in a)),verified_native_ABCD=sum(r['native_ABCD_verified'] for r in a))
for pop in ['OpenHands','AgentX']:
    a=[r for r in output[:573] if ('agentx' in r['scenario_id'])==(pop=='AgentX')]
    summary[pop]={field:sum(r[field] is True for r in a) for field in ['shared_positive_benefit','benefit_overlap','effective_budget_pressure','alternative_tradeoff_vectors','nontrivial_allocation']};summary[pop]['logical_ABC']=sum(all(r[f] is True for f in ['shared_positive_benefit','benefit_overlap','effective_budget_pressure']) for r in a)
save('summary.json',summary);save('capacity_certificates.json',cert);save('historical_fork_details.json',fork_details)
missing=[v['path'] for v in js(ROOT/'evaluation/p3_native_instrumentation_feasibility/source_verification.json')['files'] if not Path(v['path']).exists()]
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==v['sha256'] for p,v in manifest.items())
save('evidence_manifest.json',dict(status='PARTIAL',sources=manifest,missing_historical_source_paths=missing,runtime_versions={'AgentX':'SGLang 0.5.17，192槽','ExpF1':'SGLang 0.5.17，no_buffer，471槽','WP3A':'SGLang 0.5.17，extra_buffer，16槽'},script=str(Path(__file__).resolve()),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),source_unchanged=True,scope_note='600行为异质历史记录，不是600个独立Agent原生epoch；trace与配置关联见CSV和源文件。'))
print(json.dumps(summary,ensure_ascii=False,indent=2));print('历史源缺失',len(missing))
